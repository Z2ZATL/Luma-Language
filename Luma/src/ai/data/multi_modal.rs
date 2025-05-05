use std::collections::HashMap;
use std::sync::Arc;
use std::cell::RefCell;

#[derive(Debug)]
pub struct MultiModalData {
    id: i32,
    text: Option<Vec<String>>,  // Text data
    images: Option<Vec<Vec<u8>>>, // Image data (simplified as raw bytes)
    numerical: Option<Vec<Vec<f64>>>, // Numerical data
}

impl MultiModalData {
    pub fn new(id: i32) -> Self {
        MultiModalData {
            id,
            text: None,
            images: None,
            numerical: None,
        }
    }

    pub fn add_text(&mut self, text: Vec<String>) {
        self.text = Some(text);
    }

    pub fn add_images(&mut self, images: Vec<Vec<u8>>) {
        self.images = Some(images);
    }

    pub fn add_numerical(&mut self, numerical: Vec<Vec<f64>>) {
        self.numerical = Some(numerical);
    }
}

static mut MULTI_MODAL_REGISTRY: Option<Arc<RefCell<HashMap<i32, MultiModalData>>>> = None;

pub fn initialize_multi_modal_registry() {
    unsafe {
        if MULTI_MODAL_REGISTRY.is_none() {
            MULTI_MODAL_REGISTRY = Some(Arc::new(RefCell::new(HashMap::new())));
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_create_multi_modal_data(id: i32) -> i32 {
    unsafe {
        let registry = MULTI_MODAL_REGISTRY.as_ref()
            .expect("Multi-modal registry not initialized");
        let mut registry_lock = registry.borrow_mut();
        if registry_lock.contains_key(&id) {
            return -1; // Error: ID already exists
        }
        registry_lock.insert(id, MultiModalData::new(id));
        0 // Success
    }
}

#[no_mangle]
pub extern "C" fn luma_add_text_to_multi_modal(id: i32, text: *const *const std::os::raw::c_char, len: i32) -> i32 {
    if text.is_null() || len <= 0 {
        return -1;
    }
    let mut text_vec = Vec::new();
    unsafe {
        for i in 0..len {
            let c_str = *text.offset(i as isize);
            if !c_str.is_null() {
                if let Ok(s) = std::ffi::CStr::from_ptr(c_str).to_str() {
                    text_vec.push(s.to_string());
                } else {
                    return -1;
                }
            }
        }
        let registry = MULTI_MODAL_REGISTRY.as_ref()
            .expect("Multi-modal registry not initialized");
        let mut registry_lock = registry.borrow_mut();
        if let Some(data) = registry_lock.get_mut(&id) {
            data.add_text(text_vec);
            0 // Success
        } else {
            -1 // Error: ID not found
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_cleanup_multi_modal() {
    unsafe {
        if let Some(registry) = MULTI_MODAL_REGISTRY.take() {
            let mut registry_lock = registry.borrow_mut();
            registry_lock.clear();
        }
    }
}