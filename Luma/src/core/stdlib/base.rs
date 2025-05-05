pub fn print(s: &str) {
  println!("{}", s);
}

#[no_mangle]
pub extern "C" fn luma_print(s: *const std::os::raw::c_char) -> i32 {
  if s.is_null() {
      return -1;
  }
  let s_str = unsafe { std::ffi::CStr::from_ptr(s).to_str() };
  if s_str.is_err() {
      return -1;
  }
  print(s_str.unwrap());
  0
}