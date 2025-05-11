use std::collections::HashMap;
use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::data::loaders::DatasetMetadata;
use crate::ai::evaluation::evaluators;

/// Calculate absolute difference between two metrics
pub fn compare_metrics(model_a: f64, model_b: f64) -> f64 {
    (model_a - model_b).abs()
}

/// Calculate percent improvement between two metrics
pub fn percent_improvement(baseline: f64, new_model: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((new_model - baseline) / baseline) * 100.0
}

/// Model comparison structure to hold results
#[derive(Debug)]
pub struct ModelComparison {
    pub model_a_name: String,
    pub model_b_name: String,
    pub metrics_a: HashMap<String, f64>,
    pub metrics_b: HashMap<String, f64>,
    pub differences: HashMap<String, f64>,
    pub improvements: HashMap<String, f64>,
}

impl ModelComparison {
    /// Create a new comparison between two models
    pub fn new(model_a_name: &str, model_b_name: &str, 
               metrics_a: HashMap<String, f64>, metrics_b: HashMap<String, f64>) -> Self {
        let mut differences = HashMap::new();
        let mut improvements = HashMap::new();
        
        // Calculate differences and improvements for each metric
        for (metric_name, value_a) in &metrics_a {
            if let Some(value_b) = metrics_b.get(metric_name) {
                differences.insert(metric_name.clone(), compare_metrics(*value_a, *value_b));
                improvements.insert(metric_name.clone(), percent_improvement(*value_a, *value_b));
            }
        }
        
        ModelComparison {
            model_a_name: model_a_name.to_string(),
            model_b_name: model_b_name.to_string(),
            metrics_a,
            metrics_b,
            differences,
            improvements,
        }
    }
    
    /// Print comparison results
    pub fn print_comparison(&self) {
        println!("Model Comparison: '{}' vs '{}'", self.model_a_name, self.model_b_name);
        println!("+----------------+-------------+-------------+-------------+-------------+");
        println!("| Metric         | {} | {} | Difference   | Improvement  |", 
                 format!("{:<11}", self.model_a_name), 
                 format!("{:<11}", self.model_b_name));
        println!("+----------------+-------------+-------------+-------------+-------------+");
        
        // Define the metrics we want to show and their order
        let metric_order = [
            "accuracy", "precision", "recall", "f1_score", "mse", "rmse"
        ];
        
        for metric_name in metric_order.iter() {
            if let (Some(value_a), Some(value_b)) = (self.metrics_a.get(*metric_name), self.metrics_b.get(*metric_name)) {
                let diff = self.differences.get(*metric_name).unwrap_or(&0.0);
                let impr = self.improvements.get(*metric_name).unwrap_or(&0.0);
                
                println!("| {:<14} | {:<11.6} | {:<11.6} | {:<11.6} | {:<10.2}%  |", 
                         metric_name, value_a, value_b, diff, impr);
            }
        }
        
        println!("+----------------+-------------+-------------+-------------+-------------+");
    }
}

/// Compare two models using the same test dataset
pub fn compare_models(model_a: &mut NeuralNetwork, model_b: &mut NeuralNetwork, 
                      model_a_name: &str, model_b_name: &str,
                      test_dataset: &DatasetMetadata) -> ModelComparison {
    // Evaluate both models
    let metrics_a = evaluators::evaluate_trained_model(model_a, test_dataset);
    let metrics_b = evaluators::evaluate_trained_model(model_b, test_dataset);
    
    // Create and return the comparison
    ModelComparison::new(model_a_name, model_b_name, metrics_a, metrics_b)
}

#[no_mangle]
pub extern "C" fn luma_compare_models(metric_a: f64, metric_b: f64) -> f64 {
    compare_metrics(metric_a, metric_b)
}