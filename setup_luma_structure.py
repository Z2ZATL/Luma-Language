#!/usr/bin/env python3
import os
import subprocess

def main():
    """Create the basic structure for the Luma project."""
    print("Setting up Luma project structure...")
    
    # Make sure Luma directory exists
    os.makedirs("Luma", exist_ok=True)
    os.makedirs("Luma/src", exist_ok=True)
    os.makedirs("Luma/src/ai", exist_ok=True)
    os.makedirs("Luma/src/ai/engine", exist_ok=True)
    os.makedirs("Luma/src/ai/models", exist_ok=True)
    os.makedirs("Luma/src/ai/training", exist_ok=True)
    os.makedirs("Luma/src/ai/data", exist_ok=True)
    os.makedirs("Luma/src/repl", exist_ok=True)
    
    print("Luma directory structure created successfully.")
    
    # Print success message
    print("Setup complete! You can now compile and run your Luma project.")

if __name__ == "__main__":
    main()