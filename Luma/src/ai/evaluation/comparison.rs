pub fn compare_metrics(model_a: f64, model_b: f64) -> f64 {
  (model_a - model_b).abs()
}

#[no_mangle]
pub extern "C" fn luma_compare_models(metric_a: f64, metric_b: f64) -> f64 {
  compare_metrics(metric_a, metric_b)
}