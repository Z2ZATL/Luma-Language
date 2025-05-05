use std::collections::HashMap;
use std::sync::Arc;
use std::cell::RefCell;

static mut PREPROCESSED_DATA: Option<Arc<RefCell<HashMap<i32, Vec<Vec<f64>>>>>> = None;

pub fn initialize_preprocessed_registry() {
    unsafe {
        if PREPROCESSED_DATA.is_none() {
            PREPROCESSED_DATA = Some(Arc::new(RefCell::new(HashMap::new())));
        }
    }
}

pub fn normalize_data(data: &mut [Vec<f64>]) {
    for row in data.iter_mut() {
        let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
        let variance: f64 = row.iter().map(|x| (*x - mean) * (*x - mean)).sum::<f64>() / row.len() as f64;
        let std_dev = variance.sqrt();
        for val in row.iter_mut() {
            *val = (*val - mean) / std_dev;
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_preprocess_data(dataset_id: i32, data: *mut *mut f64, rows: i32, cols: i32) -> i32 {
    if data.is_null() || rows <= 0 || cols <= 0 {
        return -1;
    }
    let mut processed_data = Vec::new();
    unsafe {
        for i in 0..rows as usize {
            let row_ptr = *data.offset(i as isize);
            if !row_ptr.is_null() {
                let row = std::slice::from_raw_parts_mut(row_ptr, cols as usize);
                let mut vec_row = row.to_vec();
                normalize_data(&mut vec_row);
                processed_data.push(vec_row);
            } else {
                return -1;
            }
        }
        let registry = PREPROCESSED_DATA.as_ref()
            .expect("Preprocessed registry not initialized");
        let mut registry_lock = registry.borrow_mut();
        registry_lock.insert(dataset_id, processed_data);
        0 // Success
    }
}

#[no_mangle]
pub extern "C" fn luma_cleanup_preprocessed() {
    unsafe {
        if let Some(registry) = PREPROCESSED_DATA.take() {
            let mut registry_lock = registry.borrow_mut();
            registry_lock.clear();
        }
    }
}