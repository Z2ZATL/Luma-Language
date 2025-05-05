pub struct Callback {
  max_epochs: i32,
}

impl Callback {
  pub fn new(max_epochs: i32) -> Self {
      Callback { max_epochs }
  }

  pub fn should_stop(&self, epoch: i32) -> bool {
      epoch >= self.max_epochs
  }
}

#[no_mangle]
pub extern "C" fn luma_set_callback(max_epochs: i32) -> i32 {
  let _callback = Callback::new(max_epochs);
  0
}