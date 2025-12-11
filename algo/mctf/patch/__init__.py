# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Core patches
from .deit import apply_patch as deit
from .aug import apply_patch as aug 
from .mae import apply_patch as mae

# Optional patches
try:
    from .bert import apply_patch as bert 
except ImportError:
    bert = None

try:
    from .blip import apply_patch as blip 
except ImportError:
    blip = None

try:
    from .blip2 import apply_patch as blip2
except ImportError:
    blip2 = None

try:
    from .clip import apply_patch as clip 
except ImportError:
    clip = None

try:
    from .clip_llava import apply_patch as clip_hf
except ImportError:
    clip_hf = None

try:
    from .distilbert import apply_patch as distilbert 
except ImportError:
    distilbert = None

__all__ = ["deit", "mae", "aug", "bert", "distilbert", "blip", "blip2", "clip", "clip_hf"]
