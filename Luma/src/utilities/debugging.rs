pub fn debug_print(message: &str) {
  eprintln!("[DEBUG] {}", message);
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