use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::engine::tensor::Tensor;
use crate::ai::engine::autodiff::ComputationGraph;
use crate::ai::data::loaders::{DatasetMetadata, get_dataset};
use crate::ai::evaluation::metrics;
use std::collections::HashMap;

/// Evaluate a model using accuracy
pub fn evaluate_model(predictions: &[f64], labels: &[f64]) -> f64 {
    if predictions.len() != labels.len() {
        panic!("Length mismatch between predictions and labels");
    }
    
    metrics::accuracy(predictions, labels)
}

/// Evaluate a trained model with a test dataset
pub fn evaluate_trained_model(model: &mut NeuralNetwork, dataset: &DatasetMetadata) -> HashMap<String, f64> {
    let mut results = HashMap::new();
    let data = dataset.get_data();
    let labels = dataset.get_labels();
    
    if data.is_empty() || labels.is_empty() {
        println!("Empty dataset or labels, cannot evaluate");
        return results;
    }
    
    // Collect predictions from the model
    let mut predictions = Vec::with_capacity(data.len());
    let mut graph = ComputationGraph::new();
    
    for input in data {
        let input_tensor = Tensor::with_grad(input.clone(), vec![input.len()]);
        let input_tensor = graph.register_tensor(input_tensor);
        
        let output_tensor = model.forward(input_tensor, &mut graph);
        let output = output_tensor.get_data().to_vec();
        
        predictions.push(output[0]); // For binary classification, just use first output
    }
    
    // Extract true labels (assuming binary classification for simplicity)
    let true_labels: Vec<f64> = labels.iter()
        .map(|label| label[0])
        .collect();
    
    // Calculate various metrics
    results.insert("accuracy".to_string(), metrics::accuracy(&predictions, &true_labels));
    results.insert("precision".to_string(), metrics::precision(&predictions, &true_labels, 0.5));
    results.insert("recall".to_string(), metrics::recall(&predictions, &true_labels, 0.5));
    results.insert("f1_score".to_string(), metrics::f1_score(&predictions, &true_labels, 0.5));
    results.insert("mse".to_string(), metrics::mean_squared_error(&predictions, &true_labels));
    results.insert("rmse".to_string(), metrics::root_mean_squared_error(&predictions, &true_labels));
    
    results
}

/// Evaluate model with a named dataset
pub fn evaluate_with_dataset(model: &mut NeuralNetwork, dataset_name: &str) -> Result<HashMap<String, f64>, String> {
    match get_dataset(dataset_name) {
        Some(dataset) => Ok(evaluate_trained_model(model, &dataset)),
        None => Err(format!("Dataset '{}' not found", dataset_name))
    }
}

/// Print evaluation metrics in a nicely formatted table
pub fn print_evaluation_results(metrics: &HashMap<String, f64>) {
    println!("Evaluation Results:");
    println!("+-----------------+-------------+");
    println!("| Metric          | Value       |");
    println!("+-----------------+-------------+");
    
    // Print metrics in specific order
    let metric_order = [
        "accuracy", "precision", "recall", "f1_score", "mse", "rmse"
    ];
    
    for metric_name in metric_order.iter() {
        if let Some(value) = metrics.get(*metric_name) {
            println!("| {:<15} | {:<11.6} |", metric_name, value);
        }
    }
    
    // Print any additional metrics not in the predefined order
    for (name, value) in metrics.iter() {
        if !metric_order.contains(&name.as_str()) {
            println!("| {:<15} | {:<11.6} |", name, value);
        }
    }
    
    println!("+-----------------+-------------+");
}

/// Compatibility layer for C/FFI
#[no_mangle]
pub extern "C" fn luma_evaluate(predictions: *const f64, labels: *const f64, len: i32) -> f64 {
    if predictions.is_null() || labels.is_null() || len <= 0 {
        return -1.0;
    }
    
    unsafe {
        let pred_slice = std::slice::from_raw_parts(predictions, len as usize);
        let label_slice = std::slice::from_raw_parts(labels, len as usize);
        evaluate_model(pred_slice, label_slice)
    }
}