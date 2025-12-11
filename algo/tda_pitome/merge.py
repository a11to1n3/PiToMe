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
    if ratio >= 1.0:
        return do_nothing
    
    with torch.no_grad():
        # Handle CLS token
        if class_token:
            metric = metric[:, 1:, :]
        
        if len(metric.shape) == 2:
            metric = metric[None, ...]
        
        B, T, C = metric.shape
        
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
        
        # Only compute scores if not cached
        if topo_scores is None:
            # Initialize scorer if needed (FloodComplex by default for GPU)
            if scorer is None:
                if use_flood:
                    scorer = FloodComplexScorer()
                elif use_fast:
                    scorer = FastTopologicalScorer()
                else:
                    scorer = TopologicalScorer()
            
            # Compute topological importance scores
            # Higher score = more important = should be preserved
            topo_scores = scorer.compute_scores(metric)  # [B, T]
        
        # Normalize topo scores to [0, 1] for mixing with energy
        topo_min = topo_scores.min(dim=-1, keepdim=True)[0]
        topo_max = topo_scores.max(dim=-1, keepdim=True)[0]
        topo_norm = (topo_scores - topo_min) / (topo_max - topo_min + 1e-8)

        # Optional PiToMe energy scores for generalization
        energy_norm = None
        ew = max(0.0, min(1.0, float(energy_weight)))
        if ew > 0.0:
            metric_norm = F.normalize(metric, p=2, dim=-1)
            sim = metric_norm @ metric_norm.transpose(-1, -2)
            energy = F.elu(sim - margin, alpha=1.0).mean(dim=-1)  # high = important (PiToMe)
            e_min = energy.min(dim=-1, keepdim=True)[0]
            e_max = energy.max(dim=-1, keepdim=True)[0]
            energy_norm = (energy - e_min) / (e_max - e_min + 1e-8)

        importance = topo_norm
        if energy_norm is not None:
            importance = (1 - ew) * topo_norm + ew * energy_norm

        # Sort tokens by importance (ascending -> merge least important)
        indices = torch.argsort(importance, dim=-1, descending=False)
        
        # Split into merge candidates and protected tokens
        merge_idx = indices[:, :2*r]
        protected_idx = indices[:, 2*r:]
        
        # Split merge candidates into source and destination
        a_idx = merge_idx[:, ::2]   # Source tokens (will be merged)
        b_idx = merge_idx[:, 1::2]  # Destination tokens (receive merged content)
        
        # Compute similarity scores for merge matching
        metric_normalized = F.normalize(metric, p=2, dim=-1)
        sim = metric_normalized @ metric_normalized.transpose(-1, -2)
        
        # Get similarities between source and destination candidates
        batch_idx = torch.arange(B, device=metric.device).unsqueeze(1)
        
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
    
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        Apply token merging.
        
        Args:
            x: [B, T, C] tokens to merge
            mode: Merge mode - "mean", "sum", or "prune"
            
        Returns:
            merged: [B, T', C] merged tokens where T' < T
        """
        nonlocal r, a_idx, b_idx, dst_idx, protected_idx, batch_idx
        
        # Handle CLS token
        if class_token:
            x_cls = x[:, 0, :].unsqueeze(1)
            x = x[:, 1:, :]
        else:
            x_cls = None
        
        B, T, C = x.shape
        
        # Get source, destination, and protected tokens
        protected = x[batch_idx, protected_idx, :]
        src = x[batch_idx, a_idx, :]
        dst = x[batch_idx, b_idx, :]
        
        if mode != "prune":
            # Merge source into destination
            dst = dst.scatter_reduce(
                dim=-2,
                index=dst_idx.unsqueeze(2).expand(B, r, C),
                src=src,
                reduce=mode
            )
        
        # Concatenate: [CLS (if any)] + protected + merged destinations
        if x_cls is not None:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)
    
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
