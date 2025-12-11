
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Sys path: {sys.path}")

try:
    import algo.DiffRate.patch.mae
    print("Successfully imported algo.DiffRate.patch.mae")
except ImportError as e:
    print(f"Error importing algo.DiffRate.patch.mae: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    from algo import pitome
    print("Successfully imported algo.pitome")
except ImportError as e:
    print(f"Error importing algo.pitome: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
