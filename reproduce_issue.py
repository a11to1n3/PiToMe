
import sys
import os

# Ensure the current directory is in python path
sys.path.append(os.getcwd())

print("Attempting to import timm")
try:
    import timm
    print("Successfully imported timm")
except Exception as e:
    print(f"Failed to import timm: {e}")

try:
    print("Attempting to import algo.DiffRate.patch")
    import algo.DiffRate.patch
    print("Successfully imported algo.DiffRate.patch")
except ImportError as e:
    print(f"Caught ImportError: {e}")
except Exception as e:
    print(f"Caught Exception: {e}")

try:
    print("Attempting to import algo.pitome")
    import algo.pitome
    print("Successfully imported algo.pitome")
except Exception as e:
    print(f"Caught Exception importing algo.pitome: {e}")
