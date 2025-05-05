use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn main() {
    // Generate C bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("luma_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = Path::new(&env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");

    // Build Python bindings with pyo3
    let pyo3_status = Command::new("maturin")
        .args(&["develop", "--release"])
        .status()
        .expect("Failed to build Python bindings with maturin");

    if !pyo3_status.success() {
        panic!("Python binding build failed");
    }

    // Build WebAssembly bindings with wasm-bindgen
    let wasm_status = Command::new("wasm-bindgen")
        .args(&["--out-dir", "pkg", "--target", "web", "target/wasm32-unknown-emscripten/release/luma.wasm"])
        .status()
        .expect("Failed to generate WebAssembly bindings");

    if !wasm_status.success() {
        panic!("WebAssembly binding generation failed");
    }

    // Define error codes in a config file
    let mut config_file = File::create("src/error_codes.rs").unwrap();
    writeln!(config_file, "pub const LUMA_SUCCESS: i32 = 0;").unwrap();
    writeln!(config_file, "pub const LUMA_ERROR: i32 = -1;").unwrap();

    // Read and apply build configurations from TOML
    let native_config = std::fs::read_to_string("build/configs/native.toml").unwrap();
    let wasm_config = std::fs::read_to_string("build/configs/wasm.toml").unwrap();
    let colab_config = std::fs::read_to_string("build/configs/colab.toml").unwrap();

    let mut config_file = File::create("build_config.rs").unwrap();
    writeln!(config_file, "pub const NATIVE_CONFIG: &str = r#\"{}\"#;", native_config).unwrap();
    writeln!(config_file, "pub const WASM_CONFIG: &str = r#\"{}\"#;", wasm_config).unwrap();
    writeln!(config_file, "pub const COLAB_CONFIG: &str = r#\"{}\"#;", colab_config).unwrap();

    // Re-run build script if these files change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build/configs/native.toml");
    println!("cargo:rerun-if-changed=build/configs/wasm.toml");
    println!("cargo:rerun-if-changed=build/configs/colab.toml");
}