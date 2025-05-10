#!/usr/bin/env python3
import os
import subprocess
import sys

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
    
    # Setup Rust toolchain (if requested)
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-rust":
        try:
            print("\nSetting up Rust toolchain...")
            # Set rust to stable channel
            subprocess.run(["rustup", "default", "stable"], check=True)
            print("✅ Rust toolchain setup complete")
            
            # Check if installation was successful
            try:
                rustc_ver = subprocess.run(["rustc", "--version"], 
                                          check=True, 
                                          capture_output=True, 
                                          text=True)
                cargo_ver = subprocess.run(["cargo", "--version"], 
                                          check=True, 
                                          capture_output=True, 
                                          text=True)
                
                print(f"rustc: {rustc_ver.stdout.strip()}")
                print(f"cargo: {cargo_ver.stdout.strip()}")
            except subprocess.CalledProcessError:
                print("⚠️ Failed to verify rust installation.")
                
        except subprocess.CalledProcessError:
            print("⚠️ Failed to setup Rust toolchain. Please run 'rustup default stable' manually.")
        except FileNotFoundError:
            print("⚠️ rustup not found. Please install Rust and run 'rustup default stable' manually.")
    
    # Print success message
    print("\nSetup complete! You can now compile and run your Luma project.")
    print("To build and run the project:")
    print("  1. cd Luma")
    print("  2. cargo build")
    print("  3. cargo run -- --repl")

if __name__ == "__main__":
    main()