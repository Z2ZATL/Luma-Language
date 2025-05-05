#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn init_web() {
    println!("Initializing Luma for WebAssembly");
}

#[cfg(not(target_arch = "wasm32"))]
pub fn init_web() {
    println!("Web integration not available for non-WASM targets");
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn luma_init_web() -> i32 {
    init_web();
    0
}

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn luma_init_web() -> i32 {
    init_web();
    -1
}