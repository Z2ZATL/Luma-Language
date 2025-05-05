pub enum Optimizer {
  SGD,
  Adam,
}

#[no_mangle]
pub extern "C" fn luma_set_optimizer(model_id: i32, opt_type: i32) -> i32 {
  if opt_type != 0 && opt_type != 1 {
      return -1;
  }
  // Placeholder: Store optimizer
  0
}