pub fn to_string(x: f64) -> String {
  x.to_string()
}

#[no_mangle]
pub extern "C" fn luma_to_string(x: f64) -> *mut std::os::raw::c_char {
  let s = to_string(x);
  let c_str = std::ffi::CString::new(s).unwrap();
  c_str.into_raw()
}