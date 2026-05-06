
import os
import sys

# Try to find the log file
possible_paths = [
    r"C:\Users\IÑIGO\Documents\My Games\Mantella\logging.log",
    "logging.log"
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"Reading log from: {path}")
        try:
            with open(path, 'rb') as f:
                try:
                    f.seek(-50000, 2)
                except IOError:
                    f.seek(0)
                content = f.read().decode('utf-8', errors='ignore')
                print(content)
            break # Found and read
        except Exception as e:
            print(f"Error reading {path}: {e}")
    else:
        print(f"Path not found: {path}")
