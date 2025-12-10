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
        GPU-accelerated Flood Complex persistence computation.
        
        The flooding process simulates the growth of balls around each point.
        Components merge when balls touch, creating a persistence diagram.
        We track which points contribute to long-lived components.
        
        Args:
            points: [N, D] point tensor
            landmark_indices: [L] landmark indices
            
        Returns:
            scores: [N] persistence-based importance scores
        """
        N = points.shape[0]
        L = landmark_indices.shape[0]
        device = points.device
        
        landmarks = points[landmark_indices]
        
        # Compute distances from all points to landmarks: [N, L]
        dists_to_landmarks = torch.cdist(points, landmarks)
        
        # Assign each point to nearest landmark
        nearest_landmark_dist, nearest_landmark = dists_to_landmarks.min(dim=1)
        
        # Compute inter-landmark distances: [L, L]
        landmark_dists = torch.cdist(landmarks, landmarks)
        
        # Filtration values (discrete steps)
        filtration_values = torch.linspace(
            0, self.max_filtration, self.n_steps, device=device
        )
        
        # Initialize component labels (each landmark is its own component)
        component_labels = torch.arange(L, device=device)
        
        # Track birth times for each component
        birth_times = torch.zeros(L, device=device)
        
        # Track persistence mass for each point
        point_persistence = torch.zeros(N, device=device)
        
        # Flooding process: simulate ball growth
        for step_idx, eps in enumerate(filtration_values):
            # Find which landmark pairs are connected at this radius
            # Two landmarks connect when their distance <= 2*eps (balls touch)
            connected = landmark_dists <= (2 * eps)
            
            # Union-Find style component merging (simplified for GPU)
            # For each landmark, find minimum component label among connected landmarks
            connected_float = connected.float()
            connected_float[~connected] = float('inf')
            
            # Add self-connections
            connected_float.fill_diagonal_(0)
            
            # Propagate labels (simplified: take minimum connected label)
            for _ in range(int(math.log2(L)) + 1):  # Log iterations for convergence
                # For each component, find minimum label among connected components
                label_matrix = component_labels.unsqueeze(0).expand(L, L).float()
                label_matrix = torch.where(connected, label_matrix, torch.full_like(label_matrix, float('inf')))
                new_labels = label_matrix.min(dim=1)[0].long()
                new_labels = torch.minimum(new_labels, component_labels)
                
                if torch.equal(new_labels, component_labels):
                    break
                component_labels = new_labels
            
            # Calculate persistence contribution at this step
            # Points whose landmark is in a long-lived component get higher scores
            delta_eps = self.max_filtration / self.n_steps
            
            # Points that are within eps of their landmark are "active"
            active_points = nearest_landmark_dist <= eps
            
            # Contribution is proportional to step index (later = more important)
            point_persistence[active_points] += delta_eps
        
        # Compute final component sizes for weighting
        point_components = component_labels[nearest_landmark]
        unique_components = torch.unique(point_components)
        
        for comp in unique_components:
            mask = point_components == comp
            comp_size = mask.sum().float()
            # Larger components = more topologically significant
            point_persistence[mask] *= torch.log1p(comp_size)
        
        # Normalize to [0, 1]
        if point_persistence.max() > 0:
            point_persistence = point_persistence / point_persistence.max()
        else:
            point_persistence = torch.ones(N, device=device) / N
        
        return point_persistence
    
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

