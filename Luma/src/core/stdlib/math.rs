pub fn sqrt(x: f64) -> f64 {
  x.sqrt()
}

#[no_mangle]
pub extern "C" fn luma_sqrt(x: f64) -> f64 {
  sqrt(x)
}