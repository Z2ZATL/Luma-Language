pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
  if a.len() != b.len() {
      panic!("Vector lengths must match");
  }
  a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[no_mangle]
pub extern "C" fn luma_add_vectors(a: *const f64, b: *const f64, len: i32, out: *mut f64) -> i32 {
  if a.is_null() || b.is_null() || out.is_null() || len <= 0 {
      return -1;
  }
  unsafe {
      let a_slice = std::slice::from_raw_parts(a, len as usize);
      let b_slice = std::slice::from_raw_parts(b, len as usize);
      let result = add(a_slice, b_slice);
      std::ptr::copy_nonoverlapping(result.as_ptr(), out, len as usize);
  }
  0
}