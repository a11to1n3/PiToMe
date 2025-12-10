# Copyright (c) 2024 TDA-PiToMe
# Integration tests for DeiT with TDA-PiToMe
# --------------------------------------------------------

import sys
import pytest
import torch

sys.path.insert(0, '/Users/duypham/Documents/PiToMe')


class TestDeiTIntegration:
    """Integration tests for DeiT with TDA-PiToMe patching."""
    
    @pytest.fixture
    def patched_model(self):
        """Create a patched DeiT-Tiny model."""
        try:
            import timm
            from algo import tda_pitome
            
            # Dynamically import to handle hyphenated directory
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "tda_pitome_patch_deit",
                "/Users/duypham/Documents/PiToMe/algo/tda-pitome/patch/deit.py"
            )
            patch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_module)
            
            # Create model
            model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
            
            # Apply TDA-PiToMe patch
            patch_module.apply_patch(model, use_fast_scorer=True)
            model.ratio = 0.7
            
            return model
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_forward_pass(self, patched_model):
        """Test basic forward pass works."""
        x = torch.randn(2, 3, 224, 224)
        
        patched_model.eval()
        with torch.no_grad():
            out, flops = patched_model(x, return_flop=True)
        
        assert out.shape == (2, 1000)  # DeiT-Tiny has 1000 classes
        assert flops > 0
    
    def test_forward_without_flop(self, patched_model):
        """Test forward pass without FLOP return."""
        x = torch.randn(1, 3, 224, 224)
        
        patched_model.eval()
        with torch.no_grad():
            out = patched_model(x, return_flop=False)
        
        assert out.shape == (1, 1000)
    
    def test_different_ratios(self, patched_model):
        """Test with different merge ratios."""
        x = torch.randn(1, 3, 224, 224)
        
        patched_model.eval()
        
        for ratio in [0.5, 0.7, 0.9, 1.0]:
            patched_model.ratio = ratio
            with torch.no_grad():
                out, flops = patched_model(x, return_flop=True)
            
            assert out.shape == (1, 1000)
    
    def test_gradient_flow(self, patched_model):
        """Test gradients flow through patched model."""
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        patched_model.train()
        out, _ = patched_model(x)
        
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestFLOPReduction:
    """Tests to verify FLOP reduction with token merging."""
    
    def test_flop_decreases_with_lower_ratio(self):
        """Verify FLOPs decrease when merging more tokens."""
        try:
            import timm
            import importlib.util
            
            spec = importlib.util.spec_from_file_location(
                "tda_pitome_patch_deit",
                "/Users/duypham/Documents/PiToMe/algo/tda-pitome/patch/deit.py"
            )
            patch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_module)
            
            model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
            patch_module.apply_patch(model, use_fast_scorer=True)
            model.eval()
            
            x = torch.randn(1, 3, 224, 224)
            
            # Measure FLOPs at different ratios
            flops_dict = {}
            for ratio in [1.0, 0.8, 0.6]:
                model.ratio = ratio
                with torch.no_grad():
                    _, flops = model(x)
                flops_dict[ratio] = flops
            
            # FLOPs should decrease with lower ratio
            assert flops_dict[0.8] < flops_dict[1.0]
            assert flops_dict[0.6] < flops_dict[0.8]
            
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
