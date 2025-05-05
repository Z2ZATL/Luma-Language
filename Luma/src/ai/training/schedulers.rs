pub fn adjust_learning_rate(epoch: i32) -> f64 {
  0.01 / (1.0 + 0.001 * epoch as f64) // Simple decay
}

#[no_mangle]
pub extern "C" fn luma_adjust_learning_rate(epoch: i32) -> f64 {
  adjust_learning_rate(epoch)
}