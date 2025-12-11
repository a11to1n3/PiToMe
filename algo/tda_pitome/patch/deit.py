# Copyright (c) 2024 TDA-PiToMe
# Topological Data Analysis for Token Merging in Vision Transformers
# --------------------------------------------------------
# DeiT-specific patching for TDA-PiToMe
# --------------------------------------------------------

import math
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from .timm import TDAPiToMeAttention, TDAPiToMeBlock
from ..tda import TopologicalScorer, FastTopologicalScorer, FloodComplexScorer


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
        - Compute TDA scores once per forward pass using hooks
        """
        
        def forward(self, x, return_flop: bool = True) -> torch.Tensor:
            # Set per-layer ratios
            self._info["ratio"] = [self.ratio] * len(self.blocks)
            self._info["size"] = None
            self._info["source"] = None
            self._info["tda_scores"] = None  # Reset cached scores
            self._tda_scores_computed = False  # Flag to compute once
            self.total_flop = 0
            
            x = super().forward(x)
            
            if return_flop:
                return x, self.total_flop
            return x
        
        def _compute_tda_scores_hook(self, x: torch.Tensor):
            """Compute and cache TDA scores from input tokens."""
            if self._tda_scores_computed or self.ratio >= 1.0:
                return
            
            if hasattr(self, '_tda_scorer') and self._tda_scorer is not None:
                with torch.no_grad():
                    # Exclude CLS token for scoring
                    metric = x[:, 1:, :] if self._info["class_token"] else x
                    self._info["tda_scores"] = self._tda_scorer.compute_scores(metric)
                    self._tda_scores_computed = True
    
    return TDAPiToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, 
    trace_source: bool = False, 
    prop_attn: bool = False,
    use_fast_scorer: bool = False,
    use_flood_scorer: bool = True,  # Default to GPU-accelerated FloodComplex
    scorer_kwargs: dict = None,
    alpha: float = 0.3,  # 0.0 ≈ PiToMe density-only, 1.0 = pure persistence
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
    scorer_kwargs.setdefault("alpha", alpha)
    
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
        "tda_scores": None,  # Will be computed once per forward pass
    }
    
    # Initialize model-level TDA scorer (computed once, not per-block)
    if use_flood_scorer:
        model._tda_scorer = FloodComplexScorer(**scorer_kwargs)
    elif use_fast_scorer:
        model._tda_scorer = FastTopologicalScorer(**scorer_kwargs)
    else:
        model._tda_scorer = TopologicalScorer(**scorer_kwargs)
    
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
