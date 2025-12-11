# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

try:
    from .deit import apply_patch as deit
except ImportError as e:
    deit = None
    print(f"Warning: Could not import deit patch: {e}")

try:
    from .mae import apply_patch as mae
except ImportError as e:
    mae = None
    print(f"Warning: Could not import mae patch: {e}")

try:
    from .aug import apply_patch as aug
except ImportError as e:
    aug = None
    print(f"Warning: Could not import aug patch: {e}")

try:
    from .blip import apply_patch as blip
except ImportError as e:
    blip = None
    print(f"Warning: Could not import blip patch: {e}")

try:
    from .clip import apply_patch as clip
except ImportError as e:
    clip = None
    print(f"Warning: Could not import clip patch: {e}")

try:
    from .clip_hf import apply_patch as clip_hf
except ImportError as e:
    clip_hf = None
    print(f"Warning: Could not import clip_hf patch: {e}")

try:
    from .blip2 import apply_patch as blip2
except ImportError as e:
    blip2 = None
    print(f"Warning: Could not import blip2 patch: {e}")

__all__ = [
    "deit",
    "mae",
    "aug", 
    "blip", 
    "clip",
    "clip_hf", 
    "blip2", 
]
