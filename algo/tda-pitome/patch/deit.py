# Copyright (c) 2024 TDA-PiToMe
# Topological Data Analysis for Token Merging in Vision Transformers
# --------------------------------------------------------
# DeiT-specific patching for TDA-PiToMe
# --------------------------------------------------------

import math
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from .timm import TDAPiToMeAttention, TDAPiToMeBlock


def make_tda_pitome_class(transformer_class):
    """
    Create a TDA-PiToMe Vision Transformer class.
    
    Args:
        transformer_class: Base VisionTransformer class to extend
        
    Returns:
        TDAPiToMeVisionTransformer class
    """
    
    class TDAPiToMeVisionTransformer(transformer_class):
        """
        Vision Transformer with TDA-based token merging.
        
        Modifications:
        - Initialize ratio, token size, and token sources
        - Track FLOPs for efficiency measurement
        """
        
        def forward(self, x, return_flop: bool = True) -> torch.Tensor:
            # Set per-layer ratios
            self._info["ratio"] = [self.ratio] * len(self.blocks)
            self._info["size"] = None
            self._info["source"] = None
            self.total_flop = 0
            
            x = super().forward(x)
            
            if return_flop:
                return x, self.total_flop
            return x
        
        def forward_features(self, x):
            # Patch embedding
            x = self.patch_embed(x)
            
            # Add CLS token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), 
                    dim=1
                )
            
            # Add positional encoding
            x = self.pos_drop(x + self.pos_embed)
            
            # Process blocks with FLOP counting
            for block in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape)
                x = block(x)
            
            x = self.norm(x)
            
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            return x[:, 0], x[:, 1]
        
        def calculate_block_flop(self, shape):
            """Calculate FLOPs for a single transformer block."""
            flops = 0
            _, N, C = shape
            
            # Multi-head self-attention FLOPs
            mhsa_flops = 4 * N * C * C + 2 * N * N * C
            flops += mhsa_flops
            
            # FFN FLOPs
            ffn_flops = 8 * N * C * C
            flops += ffn_flops
            
            return flops
    
    return TDAPiToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, 
    trace_source: bool = False, 
    prop_attn: bool = False,
    use_fast_scorer: bool = False,
    use_flood_scorer: bool = True,  # Default to GPU-accelerated FloodComplex
    scorer_kwargs: dict = None,
):
    """
    Apply TDA-PiToMe patch to a Vision Transformer model.
    
    Args:
        model: VisionTransformer model to patch
        trace_source: Whether to track token sources through merging
        prop_attn: Whether to use proportional attention (unused, for compat)
        use_fast_scorer: Use FastTopologicalScorer (Witness Complex, CPU)
        use_flood_scorer: Use FloodComplexScorer (GPU-accelerated, recommended)
        scorer_kwargs: Additional kwargs for the scorer
    """
    scorer_kwargs = scorer_kwargs or {}
    
    # Create patched class
    TDAPiToMeVisionTransformer = make_tda_pitome_class(model.__class__)
    print('using', 'tda_pitome (FloodComplex GPU)' if use_flood_scorer else 'tda_pitome')
    
    # Patch the model class
    model.__class__ = TDAPiToMeVisionTransformer
    model.ratio = 1.0
    
    # Initialize info dict
    model._info = {
        "ratio": model.ratio,
        "margin": [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
    
    # Handle distillation token
    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True
    
    # Compute layer margins (for compatibility)
    num_layers = len(model.blocks)
    margins = [0.75 - 0.75 * (i / num_layers) for i in range(num_layers)]
    
    # Patch each module
    current_layer = 0
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TDAPiToMeBlock
            module.init_margin(margins[current_layer])
            module.init_scorer(
                use_fast=use_fast_scorer, 
                use_flood=use_flood_scorer, 
                **scorer_kwargs
            )
            module._info = model._info
            current_layer += 1
        elif isinstance(module, Attention):
            module.__class__ = TDAPiToMeAttention

