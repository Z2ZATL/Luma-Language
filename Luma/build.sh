#!/bin/bash
# Main build script for Luma AI/ML Framework
# Supports native, WebAssembly, and Google Colab builds

set -e

# Default values
BUILD_TYPE="native"
CLEAN_BUILD=false
RUN_TESTS=false
INSTALL=false

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build Luma AI/ML Framework for different platforms"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE       Build type: native, wasm, colab (default: native)"
    echo "  -c, --clean          Clean build (remove previous build artifacts)"
    echo "  --test               Run tests after building"
    echo "  --install            Install libraries and headers to system"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                   # Build for native platform"
    echo "  $0 -t wasm          # Build for WebAssembly"
    echo "  $0 -t colab --test  # Build for Colab with tests"
    echo "  $0 --clean --install # Clean build and install"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    native|wasm|colab)
        ;;
    *)
        echo "Error: Invalid build type '$BUILD_TYPE'"
        echo "Valid types: native, wasm, colab"
        exit 1
        ;;
esac

echo "=== Luma Build System ==="
echo "Build Type: $BUILD_TYPE"
echo "Clean Build: $CLEAN_BUILD"
echo "Run Tests: $RUN_TESTS"
echo "Install: $INSTALL"
echo "========================="

# Clean previous builds if requested
if [[ "$CLEAN_BUILD" == "true" ]]; then
    echo "Cleaning previous build artifacts..."
    rm -rf build_* target/
    cargo clean
fi

# Check prerequisites
echo "Checking prerequisites..."

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "Error: Cargo/Rust not found. Please install Rust:"
    echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check CMake installation
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake first."
    exit 1
fi

# Platform-specific prerequisite checks
case $BUILD_TYPE in
    wasm)
        if ! command -v emcc &> /dev/null; then
            echo "Error: Emscripten not found for WebAssembly build."
            echo "Please install Emscripten SDK: https://emscripten.org/"
            exit 1
        fi
        if ! rustup target list --installed | grep -q wasm32-unknown-emscripten; then
            echo "Installing WebAssembly target..."
            rustup target add wasm32-unknown-emscripten
        fi
        ;;
    colab)
        if ! python3-config --includes &> /dev/null; then
            echo "Warning: Python development headers not found."
            echo "You may need to install python3-dev package."
        fi
        ;;
esac

# Run the appropriate build script
echo "Starting $BUILD_TYPE build..."

BUILD_SCRIPT="build/build_${BUILD_TYPE}.sh"
if [[ -f "$BUILD_SCRIPT" ]]; then
    chmod +x "$BUILD_SCRIPT"
    if [[ "$RUN_TESTS" == "true" ]]; then
        "$BUILD_SCRIPT" --test
    else
        "$BUILD_SCRIPT"
    fi
else
    echo "Error: Build script not found: $BUILD_SCRIPT"
    exit 1
fi

# Install if requested
if [[ "$INSTALL" == "true" && "$BUILD_TYPE" == "native" ]]; then
    echo "Installing Luma..."
    cd build_native
    sudo cmake --install .
    echo "Installation completed!"
fi

echo ""
echo "=== Build Summary ==="
echo "✅ Build Type: $BUILD_TYPE"
echo "✅ Status: SUCCESS"

case $BUILD_TYPE in
    native)
        echo "📁 Libraries: build_native/lib/"
        echo "📁 Headers: include/"
        if [[ "$INSTALL" == "true" ]]; then
            echo "📦 System Installation: /usr/local/"
        fi
        ;;
    wasm)
        echo "📁 WASM files: build_wasm/bin/"
        echo "🌐 Ready for web deployment"
        ;;
    colab)
        echo "📁 Libraries: build_colab/lib/"
        echo "🐍 Python bindings available"
        ;;
esac

echo "===================="
echo ""
echo "Next steps:"
case $BUILD_TYPE in
    native)
        echo "  • Use pkg-config --cflags --libs luma to get compiler flags"
        echo "  • Include luma.h in your C/C++ projects"
        echo "  • Link with -lluma"
        ;;
    wasm)
        echo "  • Include luma_wasm.js in your HTML file"
        echo "  • Load LumaModule to access the API"
        echo "  • See examples in include/bindings/javascript/"
        ;;
    colab)
        echo "  • import luma in your Colab notebook"
        echo "  • Use luma.Model() to create models"
        echo "  • See examples in include/bindings/python/"
        ;;
esac

echo ""
echo "Build completed successfully! 🎉"