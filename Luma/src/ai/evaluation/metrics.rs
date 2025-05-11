/// Calculate accuracy - percentage of correct predictions
pub fn accuracy(predictions: &[f64], labels: &[f64]) -> f64 {
    let mut correct = 0;
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        if (*pred - *label).abs() < 0.5 {
            correct += 1;
        }
    }
    
    if labels.is_empty() {
        return 0.0;
    }
    
    correct as f64 / labels.len() as f64
}

/// Calculate precision - TP / (TP + FP)
pub fn precision(predictions: &[f64], labels: &[f64], threshold: f64) -> f64 {
    let mut true_positives = 0;
    let mut false_positives = 0;
    
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let predicted_class = if *pred >= threshold { 1.0 } else { 0.0 };
        
        if predicted_class == 1.0 && (*label >= 0.5) {
            true_positives += 1;
        } else if predicted_class == 1.0 && (*label < 0.5) {
            false_positives += 1;
        }
    }
    
    if true_positives + false_positives == 0 {
        return 0.0;
    }
    
    true_positives as f64 / (true_positives + false_positives) as f64
}

/// Calculate recall - TP / (TP + FN)
pub fn recall(predictions: &[f64], labels: &[f64], threshold: f64) -> f64 {
    let mut true_positives = 0;
    let mut false_negatives = 0;
    
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let predicted_class = if *pred >= threshold { 1.0 } else { 0.0 };
        
        if predicted_class == 1.0 && (*label >= 0.5) {
            true_positives += 1;
        } else if predicted_class == 0.0 && (*label >= 0.5) {
            false_negatives += 1;
        }
    }
    
    if true_positives + false_negatives == 0 {
        return 0.0;
    }
    
    true_positives as f64 / (true_positives + false_negatives) as f64
}

/// Calculate F1 score - harmonic mean of precision and recall
pub fn f1_score(predictions: &[f64], labels: &[f64], threshold: f64) -> f64 {
    let p = precision(predictions, labels, threshold);
    let r = recall(predictions, labels, threshold);
    
    if p + r == 0.0 {
        return 0.0;
    }
    
    2.0 * p * r / (p + r)
}

/// Calculate Mean Squared Error
pub fn mean_squared_error(predictions: &[f64], labels: &[f64]) -> f64 {
    if predictions.len() != labels.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let sum_squared_error: f64 = predictions.iter()
        .zip(labels.iter())
        .map(|(p, l)| (p - l).powi(2))
        .sum();
    
    sum_squared_error / predictions.len() as f64
}

/// Calculate Root Mean Squared Error
pub fn root_mean_squared_error(predictions: &[f64], labels: &[f64]) -> f64 {
    mean_squared_error(predictions, labels).sqrt()
}

/// Compatibility layer for C/FFI
#[no_mangle]
pub extern "C" fn luma_compute_accuracy(predictions: *const f64, labels: *const f64, len: i32) -> f64 {
    if predictions.is_null() || labels.is_null() || len <= 0 {
        return -1.0;
    }
    
    unsafe {
        let pred_slice = std::slice::from_raw_parts(predictions, len as usize);
        let label_slice = std::slice::from_raw_parts(labels, len as usize);
        accuracy(pred_slice, label_slice)
    }
}

/// Compatibility layer for C/FFI
#[no_mangle]
pub extern "C" fn luma_compute_f1_score(predictions: *const f64, labels: *const f64, len: i32, threshold: f64) -> f64 {
    if predictions.is_null() || labels.is_null() || len <= 0 {
        return -1.0;
    }
    
    unsafe {
        let pred_slice = std::slice::from_raw_parts(predictions, len as usize);
        let label_slice = std::slice::from_raw_parts(labels, len as usize);
        f1_score(pred_slice, label_slice, threshold)
    }
}