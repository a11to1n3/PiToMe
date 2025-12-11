# Copyright (c) 2024 TDA-PiToMe
# Topological Data Analysis for Token Merging in Vision Transformers
# --------------------------------------------------------
# TDA-based merge functions matching PiToMe interface
# --------------------------------------------------------

import math
from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F

from .tda import TopologicalScorer, FastTopologicalScorer, FloodComplexScorer


def do_nothing(x, mode=None):
    return x


def tda_pitome_vision(
    metric: torch.Tensor,
    ratio: float = 1.0,
    class_token: bool = False,
    scorer: Optional[Union[TopologicalScorer, FloodComplexScorer]] = None,
    threshold: float = 0.3,
    use_fast: bool = False,
    use_flood: bool = True,  # Default to GPU-accelerated FloodComplex
    cached_scores: Optional[torch.Tensor] = None,  # Pre-computed scores for caching
    energy_weight: float = 0.0,  # 0 = pure topology, 1 = pure PiToMe energy (superset knob)
    margin: float = 0.5,
    merge_strategy: str = "pairwise",  # "pairwise" (default, PiToMe-style) or "multiway" (n-way anchors)
    homology_dims: Optional[Union[int, tuple]] = None,  # Which homology dimensions to score (passed to scorer)
    flood_landmark_fraction: Optional[float] = 0.025,  # Reduce landmark fraction for faster FloodComplex
    flood_n_filtration_steps: Optional[int] = 20,  # Fewer filtration steps for speed
    score_proj_dim: Optional[int] = None,  # Optional random projection dim for topo scoring (speeds up cdist)
) -> Callable:
    """
    TDA-based token merging for Vision Transformers.
    
    Replaces PiToMe's energy-based scoring with Persistent Homology.
    Tokens with LOW topological scores (transient features) are merge candidates.
    Tokens with HIGH topological scores (persistent features) are preserved.
    
    Args:
        metric: [B, T, C] token embeddings (typically keys from attention)
        ratio: Target ratio of tokens to keep (0.0 to 1.0)
        class_token: Whether first token is CLS token (will be preserved)
        scorer: Pre-initialized scorer (created if None)
        threshold: Topological score threshold below which tokens are merged
        use_fast: Use FastTopologicalScorer (Witness Complex, CPU)
        use_flood: Use FloodComplexScorer (GPU-accelerated, recommended)
        cached_scores: Pre-computed topological scores [B, T] to skip expensive computation
        
    Returns:
        merge: Function that merges tokens given mode
    """
    energy_weight = max(0.0, min(1.0, float(energy_weight)))

    if ratio >= 1.0:
        return do_nothing
    
    with torch.no_grad():
        # Handle CLS token
        if class_token:
            metric = metric[:, 1:, :]
        
        if len(metric.shape) == 2:
            metric = metric[None, ...]
        
        B, T, C = metric.shape
        metric_topo = metric
        if score_proj_dim is not None and score_proj_dim < C:
            proj = torch.randn(C, score_proj_dim, device=metric.device, dtype=metric.dtype)
            metric_topo = metric @ proj
            metric_topo = F.normalize(metric_topo, p=2, dim=-1)
        
        # Calculate number of tokens to merge
        r = math.floor(T - T * ratio)
        if r <= 0:
            return do_nothing
        
        # ========================================
        # USE CACHED SCORES IF AVAILABLE (key optimization)
        # This avoids O(N²) recomputation per block
        # ========================================
        if cached_scores is not None:
            # Adapt cached scores to current token count
            # (token count decreases as we merge through layers)
            if cached_scores.shape[1] >= T:
                # Use first T scores (tokens are progressively merged)
                # Note: This is an approximation - ideally we'd track which tokens remain
                # But for speed, we use the initial ordering and truncate
                topo_scores = cached_scores[:, :T]
            else:
                # Cached scores are for fewer tokens than current (shouldn't happen normally)
                # Fall back to computing scores
                topo_scores = None
        else:
            topo_scores = None
        
        # Only compute topological scores if needed and not cached
        if topo_scores is None and energy_weight < 1.0:
            # Initialize scorer if needed (FloodComplex by default for GPU)
            if scorer is None:
                if use_flood:
                    scorer = FloodComplexScorer(
                        homology_dims=homology_dims,
                        landmark_fraction=flood_landmark_fraction or 0.05,
                        n_filtration_steps=flood_n_filtration_steps or 30,
                    )
                elif use_fast:
                    scorer = FastTopologicalScorer(homology_dims=homology_dims)
                else:
                    scorer = TopologicalScorer(homology_dims=homology_dims)
            
            # Compute topological importance scores
            # Higher score = more important = should be preserved
            topo_scores = scorer.compute_scores(metric_topo)  # [B, T]
        
        # Normalize topo scores to [0, 1] for mixing with energy
        if topo_scores is not None:
            topo_min = topo_scores.min(dim=-1, keepdim=True)[0]
            topo_max = topo_scores.max(dim=-1, keepdim=True)[0]
            topo_denom = (topo_max - topo_min).clamp_min(1e-8)
            topo_norm = (topo_scores - topo_min) / topo_denom
        else:
            topo_norm = torch.zeros((B, T), device=metric.device, dtype=metric.dtype)

        # PiToMe-style energy (density) score
        metric_normalized = F.normalize(metric, p=2, dim=-1)
        sim = metric_normalized @ metric_normalized.transpose(-1, -2)
        energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)  # high = merge candidate
        e_min = energy.min(dim=-1, keepdim=True)[0]
        e_max = energy.max(dim=-1, keepdim=True)[0]
        energy_denom = (e_max - e_min).clamp_min(1e-8)
        energy_norm = (energy - e_min) / energy_denom

        # Merge score: higher means "merge this token"
        merge_score = (1 - energy_weight) * (1.0 - topo_norm) + energy_weight * energy_norm

        # Sort tokens by merge_score (descending -> merge first)
        indices = torch.argsort(merge_score, dim=-1, descending=True)

        # ------------------------------------------------------------------ #
        # Merge pairing strategy
        # pairwise: legacy PiToMe-style bipartite pairing
        # multiway: n-way generalization – keep top tokens as anchors, merge all others to best anchor
        # ------------------------------------------------------------------ #
        merge_strategy = merge_strategy.lower()
        if merge_strategy not in ("pairwise", "multiway"):
            merge_strategy = "pairwise"

        metric_normalized = F.normalize(metric, p=2, dim=-1)
        batch_idx = torch.arange(B, device=metric.device).unsqueeze(1)

        if merge_strategy == "pairwise":
            # Split into merge candidates and protected tokens
            merge_idx = indices[:, :2*r]
            protected_idx = indices[:, 2*r:]
            
            # Split merge candidates into source and destination
            a_idx = merge_idx[:, ::2]   # Source tokens (will be merged)
            b_idx = merge_idx[:, 1::2]  # Destination tokens (receive merged content)
            
            # Gather similarity scores for merge pairs
            scores = sim.gather(
                dim=-1, 
                index=b_idx.unsqueeze(-2).expand(B, T, r)
            )
            scores = scores.gather(
                dim=-2, 
                index=a_idx.unsqueeze(-1).expand(B, r, r)
            )
            
            # Find best match for each source token
            _, dst_idx = scores.max(dim=-1)  # [B, r]
        else:
            # n-way generalization: keep top tokens as anchors, merge all others to best anchor
            keep = max(1, T - r)  # tokens to keep as anchors
            merge_tokens = T - keep
            anchors_idx = indices[:, -keep:]  # highest-importance tokens
            merge_idx = indices[:, :merge_tokens] if merge_tokens > 0 else torch.empty(B, 0, device=metric.device, dtype=torch.long)

            if merge_tokens > 0:
                anchor_feats = metric_normalized[batch_idx, anchors_idx, :]  # [B, keep, C]
                merge_feats = metric_normalized[batch_idx, merge_idx, :]     # [B, merge, C]
                sim_to_anchor = merge_feats @ anchor_feats.transpose(-1, -2)  # [B, merge, keep]
                assign_idx = sim_to_anchor.argmax(dim=-1)  # [B, merge]
            else:
                assign_idx = torch.empty(B, 0, device=metric.device, dtype=torch.long)
    
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        Apply token merging (pairwise or multiway).
        
        Args:
            x: [B, T, C] tokens to merge
            mode: Merge mode - "mean", "sum", or "prune"
            
        Returns:
            merged: [B, T', C] merged tokens where T' < T
        """
        # Handle CLS token
        if class_token:
            x_cls = x[:, 0, :].unsqueeze(1)
            x_work = x[:, 1:, :]
        else:
            x_cls = None
            x_work = x
        
        B, Tcur, C = x_work.shape

        if merge_strategy == "pairwise":
            protected = x_work[batch_idx, protected_idx, :]
            src = x_work[batch_idx, a_idx, :]
            dst = x_work[batch_idx, b_idx, :]

            if mode != "prune":
                dst = dst.scatter_reduce(
                    dim=-2,
                    index=dst_idx.unsqueeze(2).expand(B, r, C),
                    src=src,
                    reduce=mode
                )

            merged_tokens = torch.cat([protected, dst], dim=1)
        else:
            if merge_idx.numel() == 0:
                merged_tokens = x_work[batch_idx, anchors_idx, :]
            else:
                anchors = x_work[batch_idx, anchors_idx, :]  # [B, keep, C]
                merges = x_work[batch_idx, merge_idx, :]    # [B, merge, C]
                if mode != "prune":
                    anchors = anchors.scatter_reduce(
                        dim=1,
                        index=assign_idx.unsqueeze(-1).expand(B, merge_idx.shape[1], C),
                        src=merges,
                        reduce=mode,
                    )
                merged_tokens = anchors

        if x_cls is not None:
            return torch.cat([x_cls, merged_tokens], dim=1)
        return merged_tokens
    
    return merge


def tda_bsm(
    metric: torch.Tensor,
    ratio: float = 1.0,
    class_token: bool = False,
    scorer: Optional[TopologicalScorer] = None,
) -> Callable:
    """
    TDA-enhanced Bipartite Soft Matching.
    
    Uses topological scores to weight the bipartite matching,
    preferring to merge tokens with low topological importance.
    
    Args:
        metric: [B, T, C] token embeddings
        ratio: Target ratio of tokens to keep
        class_token: Whether first token is CLS token
        scorer: TopologicalScorer instance
        
    Returns:
        merge: Function that merges tokens
    """
    if ratio >= 1.0:
        return do_nothing
    
    with torch.no_grad():
        if class_token:
            metric = metric[:, 1:, :]
        
        if len(metric.shape) == 2:
            metric = metric[None, ...]
        
        B, T, C = metric.shape
        r = math.floor(T - T * ratio)
        
        if r <= 0:
            return do_nothing
        
        # Initialize scorer
        if scorer is None:
            scorer = TopologicalScorer()
        
        # Get topological scores
        topo_scores = scorer.compute_scores(metric)  # [B, T]
        
        # Normalize metric
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # Bipartite split (odd/even)
        a, b = metric[:, ::2, :], metric[:, 1::2, :]
        
        # Compute similarity
        scores = a @ b.transpose(-1, -2)
        
        # Weight by inverse topological importance
        # Tokens with low topological scores should be merged first
        topo_a = topo_scores[:, ::2]  # [B, T//2]
        topo_weights = 1.0 / (topo_a + 1e-8)  # Higher weight for low-importance tokens
        topo_weights = topo_weights / topo_weights.max(dim=-1, keepdim=True)[0]
        
        # Apply topological weighting
        scores = scores * topo_weights.unsqueeze(-1)
        
        if class_token:
            scores[:, 0, :] = -math.inf
        
        # Find best matches
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        unm_idx = edge_idx[:, r:, :]  # Unmerged
        src_idx = edge_idx[:, :r, :]  # To merge
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
    
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:, 0, :].unsqueeze(1)
            x = x[:, 1:, :]
        else:
            x_cls = None
        
        src, dst = x[:, ::2, :], x[:, 1::2, :]
        n, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_merge = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(
            dim=-2,
            index=dst_idx.expand(n, r, c),
            src=src_merge,
            reduce=mode
        )
        
        if x_cls is not None:
            return torch.cat([x_cls, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    
    return merge


# --------------------------------------------------------
# Convenience hybrid preset (PiToMe speed + slight topology)
# --------------------------------------------------------
def tda_pitome_hybrid(
    metric: torch.Tensor,
    ratio: float = 0.7,
    class_token: bool = False,
    energy_weight: float = 0.9,
    merge_strategy: str = "pairwise",
    use_flood: bool = True,
    flood_landmark_fraction: float = 0.02,
    flood_n_filtration_steps: int = 15,
    score_proj_dim: int = 32,
    homology_dims: Optional[Union[int, tuple]] = (0, 1),
    cached_scores: Optional[torch.Tensor] = None,
) -> Callable:
    """
    Fast hybrid preset leaning toward PiToMe speed with a small topological signal.
    """
    return tda_pitome_vision(
        metric=metric,
        ratio=ratio,
        class_token=class_token,
        energy_weight=energy_weight,
        merge_strategy=merge_strategy,
        use_flood=use_flood,
        flood_landmark_fraction=flood_landmark_fraction,
        flood_n_filtration_steps=flood_n_filtration_steps,
        score_proj_dim=score_proj_dim,
        homology_dims=homology_dims,
        cached_scores=cached_scores,
    )


# Merge utility functions (same as PiToMe for compatibility)
def merge_mean(merge: Callable, x: torch.Tensor) -> torch.Tensor:
    """Apply merge with mean aggregation."""
    return merge(x, mode="mean")


def merge_wavg(
    merge: Callable, 
    x: torch.Tensor, 
    size: torch.Tensor = None
) -> tuple:
    """Apply merge with weighted average based on token size."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    
    return x, size


def merge_source(
    merge: Callable, 
    x: torch.Tensor, 
    source: torch.Tensor = None
) -> torch.Tensor:
    """Track source tokens through merging."""
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    
    source = merge(source, mode="amax")
    return source


def prune(merge: Callable, x: torch.Tensor) -> torch.Tensor:
    """Apply merge as pruning (discard merged tokens)."""
    return merge(x, mode="prune")
