#!/bin/bash
# Build script for WebAssembly target

set -e

echo "Building Luma for WebAssembly..."

# Check if Emscripten is installed
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten not found. Please install Emscripten SDK first."
    echo "Visit: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Check if wasm32 target is installed
if ! rustup target list --installed | grep -q wasm32-unknown-emscripten; then
    echo "Installing wasm32-unknown-emscripten target..."
    rustup target add wasm32-unknown-emscripten
fi

# Set environment variables for WASM build
export LUMA_BUILD_WASM=1

# Create build directory
BUILD_DIR="build_wasm"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake for WASM build
emcmake cmake -DLUMA_BUILD_TYPE=wasm -DCMAKE_BUILD_TYPE=Release ../build

# Build the project
emmake make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "WebAssembly build completed successfully!"
echo "WASM files are available in: $BUILD_DIR/bin/"
echo "Include luma_wasm.js and luma_wasm.wasm in your web project"