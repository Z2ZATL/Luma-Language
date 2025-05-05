use std::collections::HashMap;

pub struct Runtime {
    variables: HashMap<String, String>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            variables: HashMap::new(),
        }
    }

    pub fn set_variable(&mut self, name: &str, value: &str) {
        self.variables.insert(name.to_string(), value.to_string());
    }

    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.variables.get(name)
    }
}

#[no_mangle]
pub extern "C" fn luma_runtime_set_var(name: *const std::os::raw::c_char, value: *const std::os::raw::c_char) -> i32 {
    if name.is_null() || value.is_null() {
        return -1;
    }
    let name_str = unsafe { std::ffi::CStr::from_ptr(name).to_str() };
    let value_str = unsafe { std::ffi::CStr::from_ptr(value).to_str() };
    if name_str.is_err() || value_str.is_err() {
        return -1;
    }
    let mut runtime = Runtime::new();
    runtime.set_variable(name_str.unwrap(), value_str.unwrap());
    0
}