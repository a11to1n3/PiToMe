# Copyright (c) 2024 TDA-PiToMe
# Topological Data Analysis for Token Merging in Vision Transformers
# --------------------------------------------------------
# TDA-PiToMe Block and Attention classes for timm models
# --------------------------------------------------------

from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block

from ..merge import tda_pitome_vision, merge_source, merge_wavg, prune
from ..tda import TopologicalScorer, FastTopologicalScorer, FloodComplexScorer


class TDAPiToMeBlock(Block):
    """
    Vision Transformer Block with TDA-based token merging.
    
    Modifications from standard Block:
    - Apply TDA token merging between attention and MLP
    - Track token sizes and optionally sources
    """
    
    def init_scorer(self, use_fast: bool = False, use_flood: bool = True, **scorer_kwargs):
        """Initialize the topological scorer. FloodComplexScorer is default (GPU-accelerated)."""
        if use_flood:
            self.scorer = FloodComplexScorer(**scorer_kwargs)
        elif use_fast:
            self.scorer = FastTopologicalScorer(**scorer_kwargs)
        else:
            self.scorer = TopologicalScorer(**scorer_kwargs)
    
    def init_margin(self, margin: float = 0.5):
        """Initialize margin for compatibility with PiToMe interface."""
        self.margin = margin
    
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)
    
    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        x_attn, metric = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)
        
        # TDA-based token merging
        ratio = self._info["ratio"].pop(0)
        
        if ratio < 1.0:
            # ========================================
            # COMPUTE TDA SCORES ONCE (on first block only)
            # This is the key optimization - compute O(N²) operation once
            # ========================================
            cached_scores = self._info.get("tda_scores", None)
            
            if cached_scores is None and hasattr(self, 'scorer') and self.scorer is not None:
                # First block: compute and cache scores
                with torch.no_grad():
                    # Use current metric (keys) for scoring
                    # Exclude CLS token if present
                    scoring_metric = metric
                    self._info["tda_scores"] = self.scorer.compute_scores(scoring_metric)
                cached_scores = self._info["tda_scores"]
            
            merge = tda_pitome_vision(
                ratio=ratio,
                metric=metric,
                class_token=self._info["class_token"],
                scorer=None,  # Don't pass scorer since we use cached scores
                cached_scores=cached_scores,
            )
            
            # Track sources if requested
            if self._info["trace_source"]:
                self._info["source"] = merge_source(
                    merge, x, self._info["source"]
                )
            
            # Apply merge with weighted averaging
            x, self._info["size"] = merge_wavg(merge, x, self._info["size"])
        
        # MLP
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        return x


class TDAPiToMeAttention(Attention):
    """
    Attention module that returns keys for TDA scoring.
    
    Modifications from standard Attention:
    - Returns mean of key vectors over heads as metric for merging
    - Supports proportional attention weighting
    """
    
    def forward(
        self, 
        x: torch.Tensor, 
        size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns attention output and key metric.
        
        Args:
            x: [B, N, C] input tokens
            size: [B, N, 1] optional token sizes for proportional attention
            
        Returns:
            x: [B, N, C] attention output
            metric: [B, N, C//num_heads] mean keys for TDA scoring
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Proportional attention based on token size
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Return keys mean as metric for TDA
        return x, k.mean(1)
