// Controls debug output verbosity
// 0 = No debug output
// 1 = Basic info (major operations, errors)
// 2 = Detailed (gradients, tensor values)
// 3 = Verbose (all operations, every tensor)
pub static DEBUG_LEVEL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

// Helper macro for conditional debug output
#[macro_export]
macro_rules! debug_print {
    ($level:expr, $($arg:tt)*) => {
        if $crate::utilities::debugging::DEBUG_LEVEL.load(std::sync::atomic::Ordering::Relaxed) >= $level {
            println!($($arg)*);
        }
    };
}

// Legacy function for backward compatibility
pub fn debug_print(message: &str) {
  eprintln!("[DEBUG] {}", message);
}

// Public function to set debug level
pub fn set_debug_level(level: usize) {
    DEBUG_LEVEL.store(level, std::sync::atomic::Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn luma_debug_print(message: *const std::os::raw::c_char) -> i32 {
  if message.is_null() {
      return -1;
  }
  let msg_str = unsafe { std::ffi::CStr::from_ptr(message).to_str() };
  if msg_str.is_err() {
      return -1;
  }
  debug_print(msg_str.unwrap());
  0
}