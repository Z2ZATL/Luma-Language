#!/bin/bash
# Build script for native platforms (Linux, macOS, Windows)

set -e

echo "Building Luma for native platform..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Cargo/Rust not found. Please install Rust first."
    exit 1
fi

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake first."
    exit 1
fi

# Create build directory
BUILD_DIR="build_native"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake for native build
cmake -DLUMA_BUILD_TYPE=native -DCMAKE_BUILD_TYPE=Release ../build

# Build the project
cmake --build . --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests if requested
if [[ "$1" == "--test" ]]; then
    echo "Running tests..."
    cd ..
    cargo test --release
fi

echo "Native build completed successfully!"
echo "Libraries are available in: $BUILD_DIR/lib/"
echo "Headers are available in: include/"