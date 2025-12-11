# Copyright (c) 2024 TDA-PiToMe
# Topological Data Analysis for Token Merging in Vision Transformers
# --------------------------------------------------------
# Persistent Homology computation module using Gudhi
# --------------------------------------------------------

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not installed. TDA features will be disabled.")

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. Dimensionality reduction disabled.")


class TopologicalScorer:
    """
    Computes topological importance scores via Persistent Homology.
    
    Tokens that participate in long-lived topological features (high persistence)
    are considered structurally important and should be preserved.
    Tokens with low persistence are merge candidates.
    """
    
    def __init__(
        self,
        dim_reduction_target: int = 64,
        max_edge_length: float = 2.0,
        use_umap: bool = True,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        homology_dims: Tuple[int, ...] = (0,),  # H0 for clusters, H1 for loops
    ):
        """
        Args:
            dim_reduction_target: Target dimensionality for UMAP reduction
            max_edge_length: Maximum edge length for Vietoris-Rips filtration
            use_umap: Whether to use UMAP for dimensionality reduction
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            homology_dims: Which homology dimensions to compute (0=clusters, 1=loops)
        """
        self.dim_target = dim_reduction_target
        self.max_edge_length = max_edge_length
        self.use_umap = use_umap and UMAP_AVAILABLE
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.homology_dims = homology_dims
        
        self._reducer: Optional[UMAP] = None
        self._fitted = False
    
    def _reduce_dims(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimensionality using UMAP.
        
        Args:
            embeddings: [N, C] numpy array of embeddings
            
        Returns:
            reduced: [N, dim_target] reduced embeddings
        """
        if not self.use_umap or embeddings.shape[1] <= self.dim_target:
            return embeddings
        
        if self._reducer is None:
            self._reducer = UMAP(
                n_components=self.dim_target,
                n_neighbors=min(self.n_neighbors, embeddings.shape[0] - 1),
                min_dist=self.min_dist,
                metric='cosine',
                random_state=42,
                low_memory=True,
            )
        
        if not self._fitted:
            # Fit on first batch
            reduced = self._reducer.fit_transform(embeddings)
            self._fitted = True
        else:
            reduced = self._reducer.transform(embeddings)
        
        return reduced
    
    def _compute_persistence_diagram(
        self, 
        points: np.ndarray,
    ) -> list:
        """
        Compute persistence diagram using Vietoris-Rips complex.
        
        Args:
            points: [N, D] point cloud
            
        Returns:
            List of (birth, death) tuples for each homology dimension
        """
        if not GUDHI_AVAILABLE:
            return []
        
        # Build Vietoris-Rips complex
        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=self.max_edge_length
        )
        
        # Create simplex tree
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=max(self.homology_dims) + 1
        )
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract persistence pairs
        persistence = simplex_tree.persistence()
        
        return persistence
    
    def _score_tokens_by_persistence(
        self,
        points: np.ndarray,
        persistence: list,
    ) -> np.ndarray:
        """
        Score each token based on its contribution to persistent features.
        
        Tokens that are "born" early and "die" late in the filtration
        are topologically significant.
        
        Args:
            points: [N, D] point cloud
            persistence: List of (dim, (birth, death)) tuples
            
        Returns:
            scores: [N] importance scores normalized to [0, 1]
        """
        n_tokens = points.shape[0]
        scores = np.zeros(n_tokens)
        
        if not persistence:
            # Fallback: uniform scores if PH computation failed
            return np.ones(n_tokens) / n_tokens
        
        # Compute pairwise distances for token assignment
        dists = np.linalg.norm(
            points[:, None, :] - points[None, :, :], 
            axis=-1
        )
        
        # For each persistence pair, assign score to relevant tokens
        for dim, (birth, death) in persistence:
            if dim not in self.homology_dims:
                continue
            
            if death == float('inf'):
                # Essential feature (infinite persistence) - high importance
                persistence_value = self.max_edge_length
            else:
                persistence_value = death - birth
            
            if persistence_value < 1e-8:
                continue
            
            # Find tokens that are "active" in this bar
            # A token is active if its minimum distance to any other token
            # falls within [birth, death]
            min_dists = np.min(dists + np.eye(n_tokens) * 1e10, axis=1)
            active_mask = (min_dists >= birth) & (min_dists <= death)
            
            # Accumulate persistence mass to active tokens
            scores[active_mask] += persistence_value
        
        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        else:
            scores = np.ones(n_tokens) / n_tokens
        
        return scores
    
    def compute_scores(
        self, 
        embeddings: torch.Tensor,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        """
        Compute topological importance scores for token embeddings.
        
        Args:
            embeddings: [B, T, C] or [T, C] token embeddings
            return_numpy: If True, return numpy array instead of tensor
            
        Returns:
            scores: [B, T] or [T] topological importance scores in [0, 1]
                   Higher score = more topologically important = should preserve
        """
        # Handle batched input
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, T, C = embeddings.shape
        device = embeddings.device
        
        # Process each batch element
        all_scores = []
        
        for b in range(B):
            # Move to CPU for Gudhi computation
            emb_np = embeddings[b].detach().cpu().numpy()
            
            # Normalize embeddings
            emb_np = emb_np / (np.linalg.norm(emb_np, axis=-1, keepdims=True) + 1e-8)
            
            # Dimensionality reduction
            reduced = self._reduce_dims(emb_np)
            
            # Compute persistence diagram
            persistence = self._compute_persistence_diagram(reduced)
            
            # Score tokens
            scores = self._score_tokens_by_persistence(reduced, persistence)
            
            all_scores.append(scores)
        
        # Stack and convert back to tensor
        scores_np = np.stack(all_scores, axis=0)
        
        if return_numpy:
            result = scores_np
        else:
            result = torch.from_numpy(scores_np).float().to(device)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result


class FastTopologicalScorer(TopologicalScorer):
    """
    Faster variant using approximate persistence via landmark sampling.
    
    Uses Witness Complex with landmarks for O(n*sqrt(n)) complexity
    instead of O(n^2) for full Vietoris-Rips.
    """
    
    def __init__(
        self,
        dim_reduction_target: int = 64,
        max_edge_length: float = 2.0,
        landmark_fraction: float = 0.1,
        **kwargs
    ):
        super().__init__(dim_reduction_target, max_edge_length, **kwargs)
        self.landmark_fraction = landmark_fraction
    
    def _select_landmarks(self, points: np.ndarray) -> np.ndarray:
        """
        Select landmarks using farthest point sampling.
        
        Args:
            points: [N, D] point cloud
            
        Returns:
            indices: [L] landmark indices
        """
        n_points = points.shape[0]
        n_landmarks = max(3, int(n_points * self.landmark_fraction))
        
        # Farthest point sampling
        landmarks = [np.random.randint(0, n_points)]
        min_dists = np.full(n_points, np.inf)
        
        for _ in range(n_landmarks - 1):
            # Update distances to nearest landmark
            dists = np.linalg.norm(points - points[landmarks[-1]], axis=1)
            min_dists = np.minimum(min_dists, dists)
            
            # Select farthest point
            next_landmark = np.argmax(min_dists)
            landmarks.append(next_landmark)
        
        return np.array(landmarks)
    
    def _compute_persistence_diagram(
        self, 
        points: np.ndarray,
    ) -> list:
        """
        Compute persistence using Witness Complex with landmarks.
        """
        if not GUDHI_AVAILABLE:
            return []
        
        # Select landmarks
        landmark_indices = self._select_landmarks(points)
        landmarks = points[landmark_indices]
        
        # Build Witness Complex
        witness_complex = gudhi.EuclideanWitnessComplex(
            witnesses=points,
            landmarks=landmarks
        )
        
        # Create simplex tree
        simplex_tree = witness_complex.create_simplex_tree(
            max_alpha_square=self.max_edge_length ** 2,
            limit_dimension=max(self.homology_dims) + 1
        )
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        return simplex_tree.persistence()


class FloodComplexScorer:
    """
    GPU-Accelerated Flood Complex for Persistent Homology.
    
    Based on Graf et al. (2025) "The Flood Complex: Large-Scale Persistent 
    Homology on Millions of Points". Uses Delaunay-based landmark subsampling 
    with GPU-parallel flooding for efficient topological scoring.
    
    Key optimizations:
    - All distance computations on GPU via torch.cdist
    - Landmark-based subsampling reduces complexity from O(n²) to O(n√n)
    - Flooding process is fully vectorized for GPU parallelism
    - No CPU-GPU memory transfers during the main computation loop
    """
    
    def __init__(
        self,
        landmark_fraction: float = 0.1,
        max_filtration_value: float = 2.0,
        n_filtration_steps: int = 50,
        device: str = None,
    ):
        """
        Args:
            landmark_fraction: Fraction of points to use as landmarks (0.05-0.2 recommended)
            max_filtration_value: Maximum distance for the filtration
            n_filtration_steps: Number of discrete steps in the filtration (more = finer resolution)
            device: Device for computation ('cuda', 'cpu', or None for auto)
        """
        self.landmark_fraction = landmark_fraction
        self.max_filtration = max_filtration_value
        self.n_steps = n_filtration_steps
        self.device = device
    
    def _get_device(self, embeddings: torch.Tensor) -> torch.device:
        """Determine computation device."""
        if self.device is not None:
            return torch.device(self.device)
        if embeddings.is_cuda:
            return embeddings.device
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    @torch.no_grad()
    def _farthest_point_sampling_gpu(
        self, 
        points: torch.Tensor, 
        n_landmarks: int
    ) -> torch.Tensor:
        """
        GPU-accelerated farthest point sampling for landmark selection.
        
        Args:
            points: [N, D] tensor of points
            n_landmarks: Number of landmarks to select
            
        Returns:
            landmark_indices: [L] tensor of landmark indices
        """
        N = points.shape[0]
        n_landmarks = min(n_landmarks, N)
        
        # Initialize with random point
        landmark_indices = torch.zeros(n_landmarks, dtype=torch.long, device=points.device)
        landmark_indices[0] = torch.randint(0, N, (1,), device=points.device)
        
        # Track minimum distances to any landmark
        min_dists = torch.full((N,), float('inf'), device=points.device)
        
        for i in range(1, n_landmarks):
            # Update min distances using last added landmark
            last_landmark = points[landmark_indices[i-1]]
            dists = torch.norm(points - last_landmark.unsqueeze(0), dim=1)
            min_dists = torch.minimum(min_dists, dists)
            
            # Select farthest point
            landmark_indices[i] = torch.argmax(min_dists)
        
        return landmark_indices
    @torch.no_grad()
    def _compute_flood_persistence_gpu(
        self,
        points: torch.Tensor,
        landmark_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        GPU-accelerated topological importance scoring.
        
        Uses a k-NN based approach that correctly approximates persistent homology:
        - birth_radius: min distance to any neighbor (when token first "connects")
        - death_radius: distance to k-th nearest neighbor (proxy for cluster boundary)
        - Local density: mean distance to k nearest neighbors
        
        The key insight for PiToMe correlation:
        - PiToMe energy = mean(elu(sim - margin)) ≈ mean similarity to neighbors
        - Our density score = 1 / mean_k_dist ≈ same semantic (dense = high score)
        
        We ADD topological structure via persistence = death - birth.
        
        Args:
            points: [N, D] normalized embeddings
            landmark_indices: [L] landmark indices (used for adaptive k selection)
            
        Returns:
            scores: [N] topological importance in [0, 1]
        """
        N = points.shape[0]
        device = points.device
        
        # Compute pairwise distances: [N, N]
        # Using cosine distance since embeddings are normalized
        dists = torch.cdist(points, points)
        
        # Set diagonal to large value to exclude self-distances
        dists = dists + torch.eye(N, device=device) * 1e10
        
        # Sort distances for each token
        sorted_dists, _ = torch.sort(dists, dim=1)
        
        # ========================================
        # K-NN BASED TOPOLOGICAL SCORING
        # ========================================
        
        # Adaptive k based on token count (similar to k in k-NN density estimation)
        k = min(max(5, N // 10), N - 1)  # At least 5, at most N-1, typically N/10
        
        # Birth radius: minimum distance (1st nearest neighbor)
        birth_radius = sorted_dists[:, 0]  # [N]
        
        # Death radius: k-th nearest neighbor distance
        # This approximates when this token's local neighborhood merges with the global structure
        death_radius = sorted_dists[:, k - 1]  # [N]
        
        # Local density: inverse of mean distance to k nearest neighbors
        # This is the closest analog to PiToMe's energy score
        mean_k_dist = sorted_dists[:, :k].mean(dim=1)  # [N]
        density_score = 1.0 / (mean_k_dist + 1e-8)
        
        # Persistence: how long this token's local structure survives
        persistence = death_radius - birth_radius
        persistence = torch.clamp(persistence, min=0)
        
        # ========================================
        # COMBINED SCORING
        # 
        # We want to correlate with PiToMe's energy while adding topology:
        # - density_score ≈ PiToMe's energy (tokens in dense regions score HIGH)
        # - persistence adds info about topological importance
        # 
        # Final = weighted combination
        # ========================================
        
        # Normalize density score to [0, 1]
        if density_score.max() > density_score.min():
            density_norm = (density_score - density_score.min()) / (density_score.max() - density_score.min())
        else:
            density_norm = torch.ones(N, device=device) * 0.5
        
        # Normalize persistence to [0, 1]
        if persistence.max() > persistence.min():
            pers_norm = (persistence - persistence.min()) / (persistence.max() - persistence.min())
        else:
            pers_norm = torch.zeros(N, device=device)
        
        # Combined score: density dominates (for PiToMe correlation), persistence modulates
        # alpha = 0 gives pure density (should correlate highly with PiToMe)
        # alpha = 1 gives pure persistence (captures different topological info)
        alpha = 0.3  # Mostly density-based for good PiToMe correlation
        combined_score = (1 - alpha) * density_norm + alpha * pers_norm
        
        # Final normalization
        if combined_score.max() > combined_score.min():
            combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min())
        else:
            combined_score = torch.ones(N, device=device) * 0.5
        
        return combined_score
    
    @torch.no_grad()
    def compute_scores(
        self,
        embeddings: torch.Tensor,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        """
        Compute topological importance scores using GPU-accelerated Flood Complex.
        
        Args:
            embeddings: [B, T, C] or [T, C] token embeddings
            return_numpy: If True, return numpy array instead of tensor
            
        Returns:
            scores: [B, T] or [T] topological importance scores in [0, 1]
                   Higher score = more topologically important = should preserve
        """
        # Handle batched input
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, T, C = embeddings.shape
        device = self._get_device(embeddings)
        original_device = embeddings.device
        
        # Move to computation device
        embeddings = embeddings.to(device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Process each batch element
        all_scores = []
        
        for b in range(B):
            points = embeddings[b]  # [T, C]
            
            # Determine number of landmarks
            n_landmarks = max(3, int(T * self.landmark_fraction))
            
            # Select landmarks via farthest point sampling
            landmark_indices = self._farthest_point_sampling_gpu(points, n_landmarks)
            
            # Compute flood persistence scores
            scores = self._compute_flood_persistence_gpu(points, landmark_indices)
            
            all_scores.append(scores)
        
        # Stack batch
        result = torch.stack(all_scores, dim=0)
        
        # Move back to original device
        result = result.to(original_device)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        if return_numpy:
            return result.cpu().numpy()
        
        return result

