
import sys
import os

sys.path.append(os.getcwd())

from src.config.config_loader import ConfigLoader

try:
    print("Loading config...")
    # Passing the default My Games path or similar
    config = ConfigLoader("C:\\Users\\IÑIGO\\Documents\\My Games\\Mantella")
    print(f"Config loaded successfully.")
    print(f"config.model (mapped to llm): '{config.llm}'")
    print(f"config.llm_api: '{config.llm_api}'")
except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()
