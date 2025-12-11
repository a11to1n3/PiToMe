# from .deit import apply_patch as deit
# from .mae  import apply_patch as mae
# import tome 
# import pitome 

__all__ = ["tome", "pitome", 'DiffRate', "tofu", "mctf", "crossget", "tda_pitome"]

PITOME = 'pitome'
TOME = 'tome'
DCT = 'dct'
TOFU = 'tofu'
LTMP = 'ltmp'
DIFFRATE = 'diffrate'
CROSSGET = 'crossget'
MCTF = 'mctf'
NONE = 'none'
TDA_PITOME = 'tda_pitome'

# Lazy imports for algorithm modules
from . import pitome, tome, dct, tofu
from .DiffRate import DiffRate

# Import TDA-PiToMe as a module with .patch attribute
try:
    from . import tda_pitome
except ImportError as e:
    tda_pitome = None
    print(f"Warning: Could not import tda_pitome: {e}")