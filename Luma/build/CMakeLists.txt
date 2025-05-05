cmake_minimum_required(VERSION 3.15)
project(Luma VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Rust support with cargo
find_program(CARGO cargo)
if(NOT CARGO)
    message(FATAL_ERROR "Cargo not found, please install Rust")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/target)

# Set build type and output directory
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add Rust target with bindings
add_custom_target(luma_rust ALL
    COMMAND ${CARGO} build --release --manifest-path ${CMAKE_SOURCE_DIR}/Cargo.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Luma with Cargo"
)

# Add C/C++ source files
set(SOURCES
    src/core/parser/lexer.cpp
    src/core/parser/parser.cpp
    src/core/interpreter/evaluator.cpp
    src/core/interpreter/runtime.cpp
    src/core/compiler/codegen.cpp
    src/ai/engine/tensor.cpp
    src/ai/data/loaders.cpp
    src/ai/models/layers.cpp
    src/ai/training/trainers.cpp
    src/ai/evaluation/metrics.cpp
    src/ai/deployment/exporters.cpp
)

# Create C/C++ executable
add_executable(luma_native ${SOURCES})
add_dependencies(luma_native luma_rust)

# Link Rust library
target_link_libraries(luma_native PRIVATE ${CMAKE_BINARY_DIR}/target/release/libluma.a)

# Find and link TensorFlow (optional for AI integration)
find_package(TensorFlow)
if(TensorFlow_FOUND)
    target_link_libraries(luma_native PRIVATE TensorFlow::TensorFlow)
    target_compile_definitions(luma_native PRIVATE USE_TENSORFLOW)
endif()

# WebAssembly build configuration
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    set_target_properties(luma_native PROPERTIES LINK_FLAGS "-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_RUNTIME_METHODS='[\"ccall\", \"cwrap\"]' -s MODULARIZE=1 -s EXPORT_NAME=\"LumaModule\"")
    set_target_properties(luma_native PROPERTIES OUTPUT_NAME "luma_wasm")
endif()

# Include bindings
set(BINDINGS_DIR ${CMAKE_SOURCE_DIR}/include/bindings)
target_sources(luma_native PRIVATE ${BINDINGS_DIR}/c.rs ${BINDINGS_DIR}/python.rs ${BINDINGS_DIR}/javascript.rs)

# Installation
install(TARGETS luma_native DESTINATION bin)

# Add custom build configurations from TOML files
configure_file(${CMAKE_SOURCE_DIR}/build/configs/native.toml ${CMAKE_BINARY_DIR}/native_config.toml COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/build/configs/wasm.toml ${CMAKE_BINARY_DIR}/wasm_config.toml COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/build/configs/colab.toml ${CMAKE_BINARY_DIR}/colab_config.toml COPYONLY)