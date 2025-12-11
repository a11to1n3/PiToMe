# Copyright (c) 2024 TDA-PiToMe
# Patch module for timm Vision Transformers

from .deit import apply_patch as deit

# timm.py contains internal classes (TDAPiToMeBlock, TDAPiToMeAttention)
# used by deit.py - no apply_patch function there
