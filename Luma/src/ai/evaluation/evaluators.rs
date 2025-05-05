pub fn evaluate_model(predictions: &[f64], labels: &[i32]) -> f64 {
  if predictions.len() != labels.len() {
      panic!("Length mismatch between predictions and labels");
  }
  let mut correct = 0;
  for (pred, label) in predictions.iter().zip(labels.iter()) {
      if (*pred - *label as f64).abs() < 0.5 { // Threshold for correctness
          correct += 1;
      }
  }
  correct as f64 / labels.len() as f64
}

#[no_mangle]
pub extern "C" fn luma_evaluate(predictions: *const f64, labels: *const i32, len: i32) -> f64 {
  if predictions.is_null() || labels.is_null() || len <= 0 {
      return -1.0;
  }
  unsafe {
      let pred_slice = std::slice::from_raw_parts(predictions, len as usize);
      let label_slice = std::slice::from_raw_parts(labels, len as usize);
      evaluate_model(pred_slice, label_slice)
  }
}