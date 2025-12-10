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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
