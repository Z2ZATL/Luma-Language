use crate::ai::data::multi_modal::{MultiModalData, initialize_multi_modal_registry};
use std::ffi::CString;

pub struct MultiModalPlugin;

impl MultiModalPlugin {
    pub fn new() -> Self {
        initialize_multi_modal_registry();
        MultiModalPlugin
    }

    pub fn add_text_data(&self, id: i32, text: Vec<&str>) -> Result<(), String> {
        let text_vec: Vec<String> = text.into_iter().map(|s| s.to_string()).collect();
        let c_strings: Vec<_> = text_vec.iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let pointers: Vec<_> = c_strings.iter().map(|c| c.as_ptr()).collect();
        let result = unsafe {
            crate::ai::data::multi_modal::luma_add_text_to_multi_modal(id, pointers.as_ptr(), text_vec.len() as i32)
        };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to add text data".to_string())
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_plugin_multi_modal_add_text(id: i32, text: *const *const std::os::raw::c_char, len: i32) -> i32 {
    if text.is_null() || len <= 0 {
        return -1;
    }
    let plugin = MultiModalPlugin::new();
    let mut text_vec = Vec::new();
    unsafe {
        for i in 0..len as isize {
            let c_str = *text.offset(i);
            if !c_str.is_null() {
                if let Ok(s) = std::ffi::CStr::from_ptr(c_str).to_str() {
                    text_vec.push(s);
                } else {
                    return -1;
                }
            }
        }
    }
    match plugin.add_text_data(id, text_vec) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}