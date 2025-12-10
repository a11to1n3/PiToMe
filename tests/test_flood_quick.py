#!/usr/bin/env python3
"""Quick test for FloodComplexScorer with small dataset."""

import sys
sys.path.insert(0, '/Users/duypham/Documents/PiToMe')

import torch
import time
import importlib.util

print("=" * 50)
print("TDA-PiToMe FloodComplexScorer Test")
print("=" * 50)

# Import TDA module
spec = importlib.util.spec_from_file_location(
    "tda", 
    "/Users/duypham/Documents/PiToMe/algo/tda-pitome/tda.py"
)
tda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tda)

print("\n✓ TDA module imported")

# Create small synthetic dataset
B, T, C = 2, 50, 64  # 2 batches, 50 tokens, 64 dims
embeddings = torch.randn(B, T, C)
print(f"✓ Test embeddings: {embeddings.shape}")

# Test FloodComplexScorer
print("\n--- FloodComplexScorer Test ---")
scorer = tda.FloodComplexScorer(
    landmark_fraction=0.15,
    max_filtration_value=2.0,
    n_filtration_steps=30
)

start = time.time()
scores = scorer.compute_scores(embeddings)
elapsed = time.time() - start

print(f"✓ Computed in {elapsed*1000:.2f}ms")
print(f"  Shape: {scores.shape}")
print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"  Sample: {scores[0, :5].tolist()}")

# Test merge function
print("\n--- Merge Function Test ---")
spec2 = importlib.util.spec_from_file_location(
    "merge",
    "/Users/duypham/Documents/PiToMe/algo/tda-pitome/merge.py"
)
merge_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(merge_mod)

merge_fn = merge_mod.tda_pitome_vision(
    metric=embeddings,
    ratio=0.7,
    use_flood=True
)

x = torch.randn(B, T, C)
merged = merge_fn(x)

print(f"✓ Original: {x.shape[1]} tokens")
print(f"✓ Merged: {merged.shape[1]} tokens")
print(f"✓ Reduction: {100*(1 - merged.shape[1]/x.shape[1]):.1f}%")

print("\n" + "=" * 50)
print("SUCCESS!")
print("=" * 50)
