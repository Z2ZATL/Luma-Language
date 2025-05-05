use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::ffi::CStr;
use std::os::raw::c_char;

// Constants for error handling
const LUMA_SUCCESS: i32 = 0;
const LUMA_ERROR: i32 = -1;

// Python module definition
#[pymodule]
fn luma(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(create_model, m)?)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(save_model, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup, m)?)?;
    Ok(())
}

#[pyfunction]
fn init() -> PyResult<()> {
    unsafe {
        luma_init();
    }
    Ok(())
}

#[pyfunction]
fn load_dataset(path: &str, name: &str, lazy: bool) -> PyResult<i32> {
    let path_c = std::ffi::CString::new(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let name_c = std::ffi::CString::new(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let lazy_flag = if lazy { 1 } else { 0 };
    let dataset_id = unsafe {
        luma_load_dataset(path_c.as_ptr(), name_c.as_ptr(), lazy_flag)
    };
    if dataset_id == LUMA_ERROR {
        return Err(PyValueError::new_err("Failed to load dataset"));
    }
    Ok(dataset_id)
}

#[pyfunction]
fn create_model(model_type: &str) -> PyResult<i32> {
    let model_type_c = std::ffi::CString::new(model_type).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let model_id = unsafe {
        luma_create_model(model_type_c.as_ptr())
    };
    if model_id == LUMA_ERROR {
        return Err(PyValueError::new_err("Failed to create model"));
    }
    Ok(model_id)
}

#[pyfunction]
fn train(model_id: i32, epochs: i32, batch_size: i32, learning_rate: f32) -> PyResult<()> {
    let result = unsafe {
        luma_train(model_id, epochs, batch_size, learning_rate)
    };
    if result != LUMA_SUCCESS {
        return Err(PyValueError::new_err("Training failed"));
    }
    Ok(())
}

#[pyfunction]
fn evaluate(model_id: i32, metrics: &str) -> PyResult<()> {
    let metrics_c = std::ffi::CString::new(metrics).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = unsafe {
        luma_evaluate(model_id, metrics_c.as_ptr())
    };
    if result != LUMA_SUCCESS {
        return Err(PyValueError::new_err("Evaluation failed"));
    }
    Ok(())
}

#[pyfunction]
fn save_model(model_id: i32, path: &str) -> PyResult<()> {
    let path_c = std::ffi::CString::new(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = unsafe {
        luma_save_model(model_id, path_c.as_ptr())
    };
    if result != LUMA_SUCCESS {
        return Err(PyValueError::new_err("Failed to save model"));
    }
    Ok(())
}

#[pyfunction]
fn cleanup() -> PyResult<()> {
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