# Makefile for Luma AI/ML Framework
# Supports multiple build targets: native, wasm, colab

.PHONY: all native wasm colab clean install test help

# Default target
all: native

# Build for native platform (Linux, macOS, Windows)
native:
	@echo "Building Luma for native platform..."
	@./build.sh --type native

# Build for WebAssembly
wasm:
	@echo "Building Luma for WebAssembly..."
	@./build.sh --type wasm

# Build for Google Colab
colab:
	@echo "Building Luma for Google Colab..."
	@./build.sh --type colab

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build_* target/
	@cargo clean

# Build and install (native only)
install: native
	@echo "Installing Luma..."
	@./build.sh --type native --install

# Run tests
test:
	@echo "Running tests..."
	@cargo test --release

# Full build with tests
test-all:
	@echo "Building and testing all targets..."
	@./build.sh --type native --test
	@./build.sh --type colab --test

# Development build (faster, with debug info)
dev:
	@echo "Building for development..."
	@cargo build

# Release build with optimizations
release:
	@echo "Building optimized release..."
	@cargo build --release

# Run REPL
repl: dev
	@echo "Starting Luma REPL..."
	@cargo run -- --repl

# Check code quality
check:
	@echo "Checking code quality..."
	@cargo check
	@cargo clippy -- -D warnings
	@cargo fmt --check

# Format code
format:
	@echo "Formatting code..."
	@cargo fmt

# Generate documentation
docs:
	@echo "Generating documentation..."
	@cargo doc --open

# Show help
help:
	@echo "Luma Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build for native platform (default)"
	@echo "  native     - Build for native platform"
	@echo "  wasm       - Build for WebAssembly"
	@echo "  colab      - Build for Google Colab"
	@echo "  clean      - Clean all build artifacts"
	@echo "  install    - Build and install (native only)"
	@echo "  test       - Run tests"
	@echo "  test-all   - Build and test all targets"
	@echo "  dev        - Development build"
	@echo "  release    - Release build"
	@echo "  repl       - Start Luma REPL"
	@echo "  check      - Check code quality"
	@echo "  format     - Format source code"
	@echo "  docs       - Generate documentation"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make                # Build for native"
	@echo "  make wasm           # Build for WebAssembly"
	@echo "  make test           # Run tests"
	@echo "  make install        # Build and install"