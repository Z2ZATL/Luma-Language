cmake_minimum_required(VERSION 3.15)
project(Luma VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Platform detection and build type configuration
set(LUMA_BUILD_TYPE "native" CACHE STRING "Build type: native, wasm, or colab")
set_property(CACHE LUMA_BUILD_TYPE PROPERTY STRINGS "native" "wasm" "colab")

message(STATUS "Building Luma for: ${LUMA_BUILD_TYPE}")

# Enable Rust support with cargo
find_program(CARGO cargo REQUIRED)
if(NOT CARGO)
    message(FATAL_ERROR "Cargo not found, please install Rust")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/target)
include_directories(${CMAKE_SOURCE_DIR}/include/bindings)

# Set build type and output directory
if(NOT CMAKE_BUILD_TYPE)
    if(LUMA_BUILD_TYPE STREQUAL "colab")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific configurations
if(LUMA_BUILD_TYPE STREQUAL "wasm")
    # WebAssembly build configuration
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "WebAssembly build requires Emscripten toolchain")
    endif()
    
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    set(LUMA_CARGO_TARGET "wasm32-unknown-emscripten")
    set(LUMA_CARGO_FLAGS "--target=${LUMA_CARGO_TARGET}")
    
    # WASM-specific linker flags
    set(WASM_LINK_FLAGS 
        "-s WASM=1"
        "-s ALLOW_MEMORY_GROWTH=1" 
        "-s EXPORTED_RUNTIME_METHODS='[\"ccall\", \"cwrap\", \"setValue\", \"getValue\"]'"
        "-s MODULARIZE=1"
        "-s EXPORT_NAME=\"LumaModule\""
        "-s EXPORTED_FUNCTIONS='[\"_main\", \"_luma_init\", \"_luma_cleanup\"]'"
        "--bind"
    )
    string(REPLACE ";" " " WASM_LINK_FLAGS_STR "${WASM_LINK_FLAGS}")
    
elseif(LUMA_BUILD_TYPE STREQUAL "colab")
    # Google Colab build configuration
    set(LUMA_CARGO_TARGET "x86_64-unknown-linux-gnu")
    set(LUMA_CARGO_FLAGS "--target=${LUMA_CARGO_TARGET}")
    
    # Find Python for Colab integration
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
    
else()
    # Native build configuration
    set(LUMA_CARGO_TARGET "")
    set(LUMA_CARGO_FLAGS "")
    
    # Detect and configure native optimizations
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    
    # OpenMP support
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, enabling parallel processing")
    endif()
    
    # CUDA support detection
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found, enabling GPU acceleration")
        enable_language(CUDA)
    endif()
endif()

# Configure Rust build based on platform
if(LUMA_CARGO_TARGET)
    set(RUST_BUILD_COMMAND ${CARGO} build --release ${LUMA_CARGO_FLAGS} --manifest-path ${CMAKE_SOURCE_DIR}/Cargo.toml)
    set(RUST_LIB_PATH ${CMAKE_SOURCE_DIR}/target/${LUMA_CARGO_TARGET}/release)
else()
    set(RUST_BUILD_COMMAND ${CARGO} build --release --manifest-path ${CMAKE_SOURCE_DIR}/Cargo.toml)
    set(RUST_LIB_PATH ${CMAKE_SOURCE_DIR}/target/release)
endif()

# Set environment variables for Rust build
set(RUST_BUILD_ENV "")
if(LUMA_BUILD_TYPE STREQUAL "wasm")
    set(RUST_BUILD_ENV ${CMAKE_COMMAND} -E env LUMA_BUILD_WASM=1)
elseif(LUMA_BUILD_TYPE STREQUAL "colab")
    set(RUST_BUILD_ENV ${CMAKE_COMMAND} -E env LUMA_BUILD_COLAB=1)
endif()

# Add Rust target with platform-specific configuration
add_custom_target(luma_rust ALL
    COMMAND ${RUST_BUILD_ENV} ${RUST_BUILD_COMMAND}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Luma Rust core for ${LUMA_BUILD_TYPE}"
    BYPRODUCTS ${RUST_LIB_PATH}/libluma.a
)

# C/C++ wrapper sources (if any exist)
file(GLOB_RECURSE C_WRAPPER_SOURCES 
    "${CMAKE_SOURCE_DIR}/src/bindings/c/*.c"
    "${CMAKE_SOURCE_DIR}/src/bindings/c/*.cpp"
)

# Create shared library target
if(LUMA_BUILD_TYPE STREQUAL "wasm")
    # For WASM, create a single executable
    add_executable(luma_wasm ${C_WRAPPER_SOURCES})
    add_dependencies(luma_wasm luma_rust)
    target_link_libraries(luma_wasm PRIVATE ${RUST_LIB_PATH}/libluma.a)
    set_target_properties(luma_wasm PROPERTIES LINK_FLAGS "${WASM_LINK_FLAGS_STR}")
    
else()
    # For native and colab, create both static and shared libraries
    add_library(luma_shared SHARED ${C_WRAPPER_SOURCES})
    add_library(luma_static STATIC ${C_WRAPPER_SOURCES})
    
    add_dependencies(luma_shared luma_rust)
    add_dependencies(luma_static luma_rust)
    
    target_link_libraries(luma_shared PRIVATE ${RUST_LIB_PATH}/libluma.a)
    target_link_libraries(luma_static PRIVATE ${RUST_LIB_PATH}/libluma.a)
    
    # Platform-specific linking
    if(LUMA_BUILD_TYPE STREQUAL "native")
        target_link_libraries(luma_shared PRIVATE pthread)
        target_link_libraries(luma_static PRIVATE pthread)
        
        if(OpenMP_CXX_FOUND)
            target_link_libraries(luma_shared PRIVATE OpenMP::OpenMP_CXX)
            target_link_libraries(luma_static PRIVATE OpenMP::OpenMP_CXX)
        endif()
        
        if(CUDAToolkit_FOUND)
            target_link_libraries(luma_shared PRIVATE CUDA::cudart)
            target_link_libraries(luma_static PRIVATE CUDA::cudart)
        endif()
        
    elseif(LUMA_BUILD_TYPE STREQUAL "colab")
        target_link_libraries(luma_shared PRIVATE ${Python3_LIBRARIES})
        target_link_libraries(luma_static PRIVATE ${Python3_LIBRARIES})
    endif()
    
    # Set library properties
    set_target_properties(luma_shared PROPERTIES
        OUTPUT_NAME "luma"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    
    set_target_properties(luma_static PROPERTIES
        OUTPUT_NAME "luma_static"
    )
endif()

# Compiler definitions based on build type
if(LUMA_BUILD_TYPE STREQUAL "wasm")
    target_compile_definitions(luma_wasm PRIVATE LUMA_BUILD_WASM=1)
elseif(LUMA_BUILD_TYPE STREQUAL "colab")
    target_compile_definitions(luma_shared PRIVATE LUMA_BUILD_COLAB=1)
    target_compile_definitions(luma_static PRIVATE LUMA_BUILD_COLAB=1)
else()
    target_compile_definitions(luma_shared PRIVATE LUMA_BUILD_NATIVE=1)
    target_compile_definitions(luma_static PRIVATE LUMA_BUILD_NATIVE=1)
    
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(luma_shared PRIVATE LUMA_OPENMP_ENABLED=1)
        target_compile_definitions(luma_static PRIVATE LUMA_OPENMP_ENABLED=1)
    endif()
    
    if(CUDAToolkit_FOUND)
        target_compile_definitions(luma_shared PRIVATE LUMA_CUDA_ENABLED=1)
        target_compile_definitions(luma_static PRIVATE LUMA_CUDA_ENABLED=1)
    endif()
endif()

# Installation configuration
if(LUMA_BUILD_TYPE STREQUAL "wasm")
    install(TARGETS luma_wasm DESTINATION bin)
    install(FILES ${CMAKE_SOURCE_DIR}/include/luma.h DESTINATION include)
    
    # Install WASM-specific files
    install(FILES 
        ${CMAKE_BINARY_DIR}/luma_wasm.wasm
        ${CMAKE_BINARY_DIR}/luma_wasm.js
        DESTINATION lib
        OPTIONAL
    )
    
else()
    install(TARGETS luma_shared luma_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(FILES ${CMAKE_SOURCE_DIR}/include/luma.h DESTINATION include)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/bindings DESTINATION include)
endif()

# Copy configuration files
configure_file(
    ${CMAKE_SOURCE_DIR}/build/configs/${LUMA_BUILD_TYPE}.toml 
    ${CMAKE_BINARY_DIR}/${LUMA_BUILD_TYPE}_config.toml 
    COPYONLY
)

# Generate pkg-config file for native builds
if(LUMA_BUILD_TYPE STREQUAL "native")
    configure_file(
        ${CMAKE_SOURCE_DIR}/build/luma.pc.in
        ${CMAKE_BINARY_DIR}/luma.pc
        @ONLY
    )
    install(FILES ${CMAKE_BINARY_DIR}/luma.pc DESTINATION lib/pkgconfig)
endif()

# Build summary
message(STATUS "")
message(STATUS "=== Luma Build Configuration ===")
message(STATUS "Build Type: ${LUMA_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target: ${LUMA_CARGO_TARGET}")
message(STATUS "Rust Library Path: ${RUST_LIB_PATH}")
if(LUMA_BUILD_TYPE STREQUAL "native")
    message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
    message(STATUS "CUDA: ${CUDAToolkit_FOUND}")
elseif(LUMA_BUILD_TYPE STREQUAL "colab")
    message(STATUS "Python: ${Python3_VERSION}")
endif()
message(STATUS "===============================")
message(STATUS "")