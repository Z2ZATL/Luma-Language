use std::sync::Arc;
use std::cell::RefCell;

#[derive(Debug)]
pub struct Accelerator {
    device: String, // e.g., "cuda", "cpu", "tpu"
    enabled: bool,
}

impl Accelerator {
    pub fn new(device: &str) -> Self {
        Accelerator {
            device: device.to_string(),
            enabled: device != "cpu",
        }
    }

    pub fn is_available(&self) -> bool {
        self.enabled
    }

    pub fn accelerate(&self, data: &[f64]) -> Vec<f64> {
        // Placeholder: Simulate acceleration
        if self.enabled {
            data.iter().map(|x| *x * 2.0).collect() // Simple doubling as acceleration example
        } else {
            data.to_vec()
        }
    }
}

static mut ACCELERATOR: Option<Arc<RefCell<Accelerator>>> = None;

pub fn initialize_accelerator(device: &str) {
    unsafe {
        if ACCELERATOR.is_none() {
            ACCELERATOR = Some(Arc::new(RefCell::new(Accelerator::new(device))));
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_init_accelerator(device: *const std::os::raw::c_char) -> i32 {
    if device.is_null() {
        return -1;
    }
    let device_str = unsafe { std::ffi::CStr::from_ptr(device).to_str() };
    if device_str.is_err() {
        return -1;
    }
    initialize_accelerator(device_str.unwrap());
    0
}

#[no_mangle]
pub extern "C" fn luma_cleanup_accelerator() {
    unsafe { ACCELERATOR.take(); }
}