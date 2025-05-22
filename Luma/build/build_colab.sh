#!/bin/bash
# Build script for Google Colab environment

set -e

echo "Building Luma for Google Colab..."

# Check if Python development headers are available
if ! python3-config --includes &> /dev/null; then
    echo "Warning: Python development headers not found. Installing python3-dev..."
    apt-get update && apt-get install -y python3-dev python3-pip
fi

# Check if required Python packages are installed
python3 -c "import numpy, pandas" 2>/dev/null || {
    echo "Installing required Python packages..."
    pip3 install numpy pandas matplotlib
}

# Set environment variables for Colab build
export LUMA_BUILD_COLAB=1

# Install maturin for Python bindings if not available
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin for Python bindings..."
    pip3 install maturin
fi

# Create build directory
BUILD_DIR="build_colab"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake for Colab build
cmake -DLUMA_BUILD_TYPE=colab -DCMAKE_BUILD_TYPE=Debug ../build

# Build the project
cmake --build . --parallel $(nproc 2>/dev/null || echo 4)

# Build Python bindings
cd ..
maturin develop --release

echo "Google Colab build completed successfully!"
echo "Python bindings are available for import in Colab notebooks"
echo "Example usage:"
echo "  import luma"
echo "  model = luma.Model('neural_network')"