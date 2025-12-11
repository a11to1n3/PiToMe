# Copyright (c) 2024 TDA-PiToMe
# Unit tests for TDA module
# --------------------------------------------------------

import sys
import pytest
import numpy as np
import torch

# Add algo to path for imports
sys.path.insert(0, '/Users/duypham/Documents/PiToMe')


class TestTopologicalScorerMock:
    """
    Tests for TopologicalScorer using mock data.
    Runs without gudhi/umap to test basic structure.
    """
    
    def test_import_tda_module(self):
        """Verify tda module can be imported."""
        from algo import tda_pitome
        assert hasattr(tda_pitome, 'TopologicalScorer')
        assert hasattr(tda_pitome, 'tda_pitome_vision')
    
    def test_scorer_instantiation(self):
        """Verify TopologicalScorer can be instantiated."""
        try:
            from algo import tda_pitome
            scorer = tda_pitome.TopologicalScorer(
                dim_reduction_target=32,
                max_edge_length=2.0
            )
            assert scorer.dim_target == 32
            assert scorer.max_edge_length == 2.0
        except ImportError as e:
            pytest.skip(f"TDA dependencies not installed: {e}")


class TestTopologicalScorerFull:
    """
    Full tests requiring gudhi and umap-learn.
    """
    
    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        try:
            from algo import tda_pitome
            return tda_pitome.TopologicalScorer(
                dim_reduction_target=16,
                max_edge_length=2.0,
                use_umap=False,  # Skip UMAP for faster tests
            )
        except ImportError:
            pytest.skip("gudhi not installed")
    
    def test_compute_scores_shape_2d(self, scorer):
        """Test scorer output shape for 2D input."""
        embeddings = torch.randn(50, 64)  # 50 tokens, 64 dims
        scores = scorer.compute_scores(embeddings)
        
        assert scores.shape == (50,)
        assert scores.dtype == torch.float32
    
    def test_compute_scores_shape_3d(self, scorer):
        """Test scorer output shape for batched 3D input."""
        embeddings = torch.randn(2, 50, 64)  # Batch of 2
        scores = scorer.compute_scores(embeddings)
        
        assert scores.shape == (2, 50)
    
    def test_scores_normalized(self, scorer):
        """Verify scores are in [0, 1] range."""
        embeddings = torch.randn(30, 64)
        scores = scorer.compute_scores(embeddings)
        
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
    
    def test_scores_deterministic(self, scorer):
        """Verify scores are deterministic for same input."""
        torch.manual_seed(42)
        embeddings = torch.randn(30, 64)
        
        scores1 = scorer.compute_scores(embeddings)
        scores2 = scorer.compute_scores(embeddings)
        
        assert torch.allclose(scores1, scores2)


class TestTDAPiToMeVision:
    """Tests for tda_pitome_vision merge function."""
    
    @pytest.fixture
    def merge_fn(self):
        """Create merge function."""
        try:
            from algo import tda_pitome
            metric = torch.randn(2, 100, 64)  # 2 batches, 100 tokens
            return tda_pitome.tda_pitome_vision(
                metric=metric,
                ratio=0.7,
                class_token=False,
            )
        except ImportError:
            pytest.skip("TDA dependencies not installed")
    
    def test_merge_reduces_tokens(self, merge_fn):
        """Verify merge reduces token count."""
        x = torch.randn(2, 100, 64)
        merged = merge_fn(x, mode="mean")
        
        # Should have fewer tokens
        assert merged.shape[1] < x.shape[1]
        assert merged.shape[0] == x.shape[0]  # Batch preserved
        assert merged.shape[2] == x.shape[2]  # Channels preserved
    
    def test_merge_with_class_token(self):
        """Verify CLS token is preserved."""
        try:
            from algo import tda_pitome
            
            metric = torch.randn(2, 100, 64)
            merge = tda_pitome.tda_pitome_vision(
                metric=metric,
                ratio=0.7,
                class_token=True,
            )
            
            x = torch.randn(2, 100, 64)
            x[:, 0, :] = 999.0  # Mark CLS token
            
            merged = merge(x, mode="mean")
            
            # CLS token should be preserved
            assert torch.allclose(merged[:, 0, :], x[:, 0, :])
        except ImportError:
            pytest.skip("TDA dependencies not installed")
    
    def test_merge_mode_prune(self, merge_fn):
        """Test prune mode discards tokens."""
        x = torch.randn(2, 100, 64)
        merged = merge_fn(x, mode="prune")
        
        assert merged.shape[1] < x.shape[1]


class TestMergeUtilities:
    """Tests for merge utility functions."""
    
    def test_merge_wavg(self):
        """Test weighted average merge."""
        try:
            from algo import tda_pitome
            from algo import tda_pitome as tdp
            
            metric = torch.randn(2, 100, 64)
            merge = tdp.tda_pitome_vision(metric=metric, ratio=0.7)
            
            x = torch.randn(2, 100, 64)
            size = torch.ones(2, 100, 1)
            
            merged_x, merged_size = tdp.merge.merge_wavg(merge, x, size)
            
            assert merged_x.shape[1] == merged_size.shape[1]
            assert merged_size.sum() == size.sum()  # Total size preserved
        except ImportError:
            pytest.skip("TDA dependencies not installed")


class TestTDAGeneralizationTheorem:
    """Tests for theoretical guarantees: PiToMe reduction and topological refinement."""

    def test_reduction_matches_pitome_pairwise(self):
        """energy_weight=1 should recover PiToMe pairwise merge ordering/output."""
        torch.manual_seed(0)
        try:
            from algo import pitome, tda_pitome
        except ImportError:
            pytest.skip("Algo modules not available")
        if tda_pitome is None:
            pytest.skip("tda_pitome module unavailable")

        metric = torch.randn(1, 8, 16)
        ratio = 0.5

        merge_tda = tda_pitome.tda_pitome_vision(
            metric=metric,
            ratio=ratio,
            class_token=False,
            energy_weight=1.0,
            merge_strategy="pairwise",
            use_flood=False,
            use_fast=False,
        )
        merge_pitome = pitome.merge.pitome_vision(
            metric=metric,
            ratio=ratio,
            class_token=False,
        )

        # Token identities to detect ordering and aggregation
        x = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        merged_tda = merge_tda(x.clone(), mode="mean")
        merged_pitome = merge_pitome(x.clone(), mode="mean")

        assert merged_tda.shape == merged_pitome.shape
        assert torch.allclose(merged_tda, merged_pitome, atol=1e-6)

    def test_topology_refines_when_energy_flat(self):
        """
        When energy is flat but topology varies, alpha<1 should change ordering:
        multiway+prune should drop the lowest persistence token.
        """
        try:
            from algo import tda_pitome
        except ImportError:
            pytest.skip("tda_pitome module unavailable")
        if tda_pitome is None:
            pytest.skip("tda_pitome module unavailable")

        class DummyScorer:
            def __init__(self, scores: torch.Tensor):
                self.scores = scores

            def compute_scores(self, embeddings, return_numpy: bool = False):
                out = self.scores
                if embeddings.dim() == 3:
                    out = out.unsqueeze(0).expand(embeddings.shape[0], -1)
                if return_numpy:
                    return out.cpu().numpy()
                return out.to(embeddings.device)

        # Flat energy: identical tokens; topology: distinct persistence scores
        topo_scores = torch.tensor([0.9, 0.1, 0.5, 0.4, 0.2], dtype=torch.float32)
        scorer = DummyScorer(topo_scores)
        metric = torch.ones(1, 5, 4)  # identical vectors -> flat density

        merge = tda_pitome.tda_pitome_vision(
            metric=metric,
            ratio=0.8,  # keep 4, merge 1
            class_token=False,
            scorer=scorer,
            energy_weight=0.0,  # pure topology
            merge_strategy="multiway",
            use_flood=False,
            use_fast=False,
        )

        x = torch.eye(5).unsqueeze(0)
        merged = merge(x, mode="prune")  # anchors only
        kept_indices = torch.argmax(merged[0], dim=-1).tolist()

        # Lowest persistence token (index 1 with score 0.1) should be removed
        assert 1 not in kept_indices
        assert set(kept_indices) == {0, 2, 3, 4}


class TestHybridPreset:
    """Tests for the hybrid fast preset."""

    def test_hybrid_returns_callable_and_reduces_tokens(self):
        torch.manual_seed(0)
        try:
            from algo import tda_pitome
        except ImportError:
            pytest.skip("tda_pitome module unavailable")
        if tda_pitome is None:
            pytest.skip("tda_pitome module unavailable")

        metric = torch.randn(1, 10, 8)
        merge = tda_pitome.tda_pitome_hybrid(metric=metric, ratio=0.8)
        x = torch.randn(1, 10, 8)
        merged = merge(x)
        assert merged.shape[1] < x.shape[1]
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
