use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Detect target platform and build configuration
    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
    let build_type = detect_build_type(&target);
    
    println!("Building Luma for target: {} ({})", target, build_type);

    // Create output directories
    create_output_directories();
    
    // Generate build configuration constants
    generate_build_constants(&build_type);
    
    // Generate error codes
    generate_error_codes();
    
    // Build based on target platform
    match build_type.as_str() {
        "native" => build_native(),
        "wasm" => build_wasm(),
        "colab" => build_colab(),
        _ => build_native(), // Default to native
    }
    
    // Generate C bindings if needed
    if should_generate_c_bindings(&build_type) {
        generate_c_bindings();
    }
    
    // Set up cargo features based on build type
    setup_cargo_features(&build_type);
    
    println!("Build configuration complete for {}", build_type);
}

fn detect_build_type(target: &str) -> String {
    // Check environment variables first
    if env::var("LUMA_BUILD_COLAB").is_ok() {
        return "colab".to_string();
    }
    
    if env::var("LUMA_BUILD_WASM").is_ok() {
        return "wasm".to_string();
    }
    
    // Detect based on target triple
    if target.contains("wasm32") {
        "wasm".to_string()
    } else if target.contains("emscripten") {
        "wasm".to_string()
    } else {
        "native".to_string()
    }
}

fn create_output_directories() {
    let dirs = ["target/include", "target/bindings", "target/pkg"];
    for dir in &dirs {
        if let Err(e) = fs::create_dir_all(dir) {
            println!("cargo:warning=Failed to create directory {}: {}", dir, e);
        }
    }
}

fn generate_build_constants(build_type: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("build_config.rs");
    let mut f = File::create(&dest_path).unwrap();

    writeln!(f, "// Auto-generated build configuration").unwrap();
    writeln!(f, "pub const BUILD_TYPE: &str = \"{}\";", build_type).unwrap();
    writeln!(f, "pub const BUILD_TARGET: &str = \"{}\";", env::var("TARGET").unwrap_or_default()).unwrap();
    writeln!(f, "pub const BUILD_TIMESTAMP: &str = \"{}\";", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")).unwrap();

    // Load and embed configuration files
    if let Ok(config_content) = load_config_for_build_type(build_type) {
        writeln!(f, "pub const BUILD_CONFIG: &str = r#\"{}\"#;", config_content).unwrap();
    }
}

fn generate_error_codes() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("error_codes.rs");
    let mut f = File::create(&dest_path).unwrap();

    writeln!(f, "// Auto-generated error codes").unwrap();
    writeln!(f, "pub const LUMA_SUCCESS: i32 = 0;").unwrap();
    writeln!(f, "pub const LUMA_ERROR: i32 = -1;").unwrap();
    writeln!(f, "pub const LUMA_ERROR_INVALID_INPUT: i32 = -2;").unwrap();
    writeln!(f, "pub const LUMA_ERROR_OUT_OF_MEMORY: i32 = -3;").unwrap();
    writeln!(f, "pub const LUMA_ERROR_NOT_IMPLEMENTED: i32 = -4;").unwrap();
    writeln!(f, "pub const LUMA_ERROR_CUDA_ERROR: i32 = -5;").unwrap();
}

fn load_config_for_build_type(build_type: &str) -> Result<String, std::io::Error> {
    let config_path = format!("build/configs/{}.toml", build_type);
    fs::read_to_string(&config_path)
}

fn build_native() {
    println!("Building native version with optimizations");
    
    // Link native libraries if available
    println!("cargo:rustc-link-lib=dylib=pthread");
    
    // Add native-specific features
    println!("cargo:rustc-cfg=feature=\"native\"");
    
    // Check for CUDA support
    if Command::new("nvcc").arg("--version").output().is_ok() {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
    
    // Check for OpenMP support
    if Path::new("/usr/lib/x86_64-linux-gnu/libomp.so").exists() {
        println!("cargo:rustc-link-lib=dylib=omp");
        println!("cargo:rustc-cfg=feature=\"openmp\"");
    }
}

fn build_wasm() {
    println!("Building WebAssembly version");
    
    // WASM-specific configurations
    println!("cargo:rustc-cfg=feature=\"wasm\"");
    println!("cargo:rustc-cfg=target_family=\"wasm\"");
    
    // Generate WASM bindings if wasm-bindgen is available
    if Command::new("wasm-bindgen").arg("--version").output().is_ok() {
        println!("wasm-bindgen detected, will generate JS bindings");
    } else {
        println!("cargo:warning=wasm-bindgen not found, JS bindings will not be generated");
    }
}

fn build_colab() {
    println!("Building Colab-compatible version");
    
    // Colab-specific configurations
    println!("cargo:rustc-cfg=feature=\"colab\"");
    println!("cargo:rustc-cfg=feature=\"python-bindings\"");
    
    // Check for Python development headers
    if let Ok(python_config) = Command::new("python3-config").arg("--includes").output() {
        let includes = String::from_utf8_lossy(&python_config.stdout);
        for include in includes.split_whitespace() {
            if include.starts_with("-I") {
                println!("cargo:rustc-link-search=native={}", &include[2..]);
            }
        }
    }
}

fn should_generate_c_bindings(build_type: &str) -> bool {
    matches!(build_type, "native" | "colab")
}

fn generate_c_bindings() {
    // Only generate if bindgen is available and header exists
    if !Path::new("include/luma.h").exists() {
        println!("cargo:warning=include/luma.h not found, skipping C bindings generation");
        return;
    }
    
    if Command::new("bindgen").arg("--version").output().is_err() {
        println!("cargo:warning=bindgen not found, C bindings will not be auto-generated");
        return;
    }
    
    println!("Generating C bindings...");
    
    // Create a wrapper header that includes all necessary headers
    let wrapper_content = r#"
#include "include/luma.h"
// Additional C API functions can be declared here
"#;
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let wrapper_path = Path::new(&out_dir).join("wrapper.h");
    fs::write(&wrapper_path, wrapper_content).expect("Failed to write wrapper.h");
    
    // Use cbindgen to generate bindings from Rust code
    if let Ok(crate_dir) = env::var("CARGO_MANIFEST_DIR") {
        let config = cbindgen::Config {
            header: Some("/* Auto-generated C bindings for Luma */".to_string()),
            include_guard: Some("LUMA_BINDINGS_H".to_string()),
            language: cbindgen::Language::C,
            ..Default::default()
        };
        
        if let Ok(bindings) = cbindgen::generate_with_config(crate_dir, config) {
            let bindings_path = Path::new("target/include/luma_bindings.h");
            bindings.write_to_file(bindings_path);
            println!("C bindings generated at target/include/luma_bindings.h");
        }
    }
}

fn setup_cargo_features(build_type: &str) {
    match build_type {
        "native" => {
            println!("cargo:rustc-cfg=feature=\"native\"");
            println!("cargo:rustc-cfg=feature=\"parallel\"");
        },
        "wasm" => {
            println!("cargo:rustc-cfg=feature=\"wasm\"");
            println!("cargo:rustc-cfg=feature=\"web\"");
        },
        "colab" => {
            println!("cargo:rustc-cfg=feature=\"colab\"");
            println!("cargo:rustc-cfg=feature=\"python\"");
        },
        _ => {}
    }
    
    // Common features
    println!("cargo:rustc-cfg=feature=\"json\"");
    
    // Re-run build script if these files change
    println!("cargo:rerun-if-changed=include/luma.h");
    println!("cargo:rerun-if-changed=build/configs/native.toml");
    println!("cargo:rerun-if-changed=build/configs/wasm.toml");
    println!("cargo:rerun-if-changed=build/configs/colab.toml");
}