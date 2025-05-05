use wasm_bindgen::prelude::*;
use std::ffi::CStr;
use std::os::raw::c_char;

// Constants for error handling
const LUMA_SUCCESS: i32 = 0;
const LUMA_ERROR: i32 = -1;

#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
    unsafe {
        luma_init();
    }
    Ok(())
}

#[wasm_bindgen]
pub fn load_dataset(path: &str, name: &str, lazy: bool) -> Result<i32, JsValue> {
    let path_c = std::ffi::CString::new(path).map_err(|_| JsValue::from_str("Invalid path"))?;
    let name_c = std::ffi::CString::new(name).map_err(|_| JsValue::from_str("Invalid name"))?;
    let lazy_flag = if lazy { 1 } else { 0 };
    let dataset_id = unsafe {
        luma_load_dataset(path_c.as_ptr(), name_c.as_ptr(), lazy_flag)
    };
    if dataset_id == LUMA_ERROR {
        return Err(JsValue::from_str("Failed to load dataset"));
    }
    Ok(dataset_id)
}

#[wasm_bindgen]
pub fn create_model(model_type: &str) -> Result<i32, JsValue> {
    let model_type_c = std::ffi::CString::new(model_type).map_err(|_| JsValue::from_str("Invalid model type"))?;
    let model_id = unsafe {
        luma_create_model(model_type_c.as_ptr())
    };
    if model_id == LUMA_ERROR {
        return Err(JsValue::from_str("Failed to create model"));
    }
    Ok(model_id)
}

#[wasm_bindgen]
pub fn train(model_id: i32, epochs: i32, batch_size: i32, learning_rate: f32) -> Result<(), JsValue> {
    let result = unsafe {
        luma_train(model_id, epochs, batch_size, learning_rate)
    };
    if result != LUMA_SUCCESS {
        return Err(JsValue::from_str("Training failed"));
    }
    Ok(())
}

#[wasm_bindgen]
pub fn evaluate(model_id: i32, metrics: &str) -> Result<(), JsValue> {
    let metrics_c = std::ffi::CString::new(metrics).map_err(|_| JsValue::from_str("Invalid metrics"))?;
    let result = unsafe {
        luma_evaluate(model_id, metrics_c.as_ptr())
    };
    if result != LUMA_SUCCESS {
        return Err(JsValue::from_str("Evaluation failed"));
    }
    Ok(())
}

#[wasm_bindgen]
pub fn save_model(model_id: i32, path: &str) -> Result<(), JsValue> {
    let path_c = std::ffi::CString::new(path).map_err(|_| JsValue::from_str("Invalid path"))?;
    let result = unsafe {
        luma_save_model(model_id, path_c.as_ptr())
    };
    if result != LUMA_SUCCESS {
        return Err(JsValue::from_str("Failed to save model"));
    }
    Ok(())
}

#[wasm_bindgen]
pub fn cleanup() -> Result<(), JsValue> {
    unsafe {
        luma_cleanup();
    }
    Ok(())
}

// Declare external C functions
extern "C" {
    fn luma_init();
    fn luma_load_dataset(path: *const c_char, name: *const c_char, lazy: i32) -> i32;
    fn luma_create_model(model_type: *const c_char) -> i32;
    fn luma_train(model_id: i32, epochs: i32, batch_size: i32, learning_rate: f32) -> i32;
    fn luma_evaluate(model_id: i32, metrics: *const c_char) -> i32;
    fn luma_save_model(model_id: i32, path: *const c_char) -> i32;
    fn luma_cleanup();
}