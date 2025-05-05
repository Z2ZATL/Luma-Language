pub struct Layer {
  id: i32,
  neurons: i32,
}

impl Layer {
  pub fn new(id: i32, neurons: i32) -> Self {
      Layer { id, neurons }
  }
}

#[no_mangle]
pub extern "C" fn luma_add_layer(model_id: i32, neurons: i32) -> i32 {
  let _layer = Layer::new(model_id, neurons);
  0
}