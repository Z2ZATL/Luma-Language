//! Luma - A Domain-Specific Language for AI Development
//!
//! This library provides the core functionality for Luma, including AI pipelines,
//! integrations, utilities, and plugins.

pub mod ai;
pub mod core;
pub mod compiler;
pub mod integrations;
pub mod utilities;

#[path = "../plugins/mod.rs"]
pub mod plugins;

pub mod repl;

pub use ai::data as ai_data;
pub use ai::deployment as ai_deployment;
pub use ai::engine as ai_engine;
pub use ai::evaluation as ai_evaluation;
pub use ai::models as ai_models;
pub use ai::training as ai_training;

pub use core::compiler as core_compiler;
pub use core::interpreter as core_interpreter;
pub use core::parser as core_parser;
pub use core::stdlib as core_stdlib;

pub use integrations::tensorflow as int_tensorflow;
pub use integrations::pytorch as int_pytorch;
pub use integrations::huggingface as int_huggingface;
pub use integrations::web as int_web;

pub use utilities::profiling as util_profiling;
pub use utilities::debugging as util_debugging;
pub use utilities::logging as util_logging;
pub use utilities::visualization as util_visualization;

// Community plugins support has been temporarily removed
// pub use plugins::community as plg_community;
pub use plugins::registry as plg_registry;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    // Data loading tests
    mod data_tests {
        use crate::ai::data::loaders;
        
        #[test]
        fn test_load_dataset() {
            // Test loading a dataset with default parameters
            let result = loaders::load_dataset("samples/iris.csv", "test_iris", false, None);
            assert!(result.is_ok());
        }
        
        #[test]
        fn test_load_multiple_datasets() {
            // Load first dataset
            let result1 = loaders::load_dataset("samples/iris.csv", "test_iris1", false, None);
            assert!(result1.is_ok());
            
            // Load second dataset
            let result2 = loaders::load_dataset("samples/data.csv", "test_data1", false, None);
            assert!(result2.is_ok());
        }
        
        #[test]
        fn test_load_with_label_column() {
            // Test loading with specific label column
            let result = loaders::load_dataset("samples/iris.csv", "test_iris_labels", false, Some(4));
            assert!(result.is_ok());
        }
        
        #[test]
        fn test_lazy_loading() {
            // Test lazy loading
            let result = loaders::load_dataset("samples/iris.csv", "test_iris_lazy", true, None);
            assert!(result.is_ok());
        }
    }
    
    // Model creation tests
    mod model_tests {
        use crate::ai::models::advanced::NeuralNetwork;
        use crate::ai::models::layers::Layer;
        
        #[test]
        fn test_create_neural_network() {
            // Test creating a new neural network with specific layer sizes
            let nn = NeuralNetwork::new("test_model", vec![784, 128, 64, 10]);
            
            // Check the network structure
            assert_eq!(nn.id, "test_model");
            assert_eq!(nn.layers.len(), 3); // Should have 3 layers (input->hidden, hidden->hidden, hidden->output)
        }
        
        #[test]
        fn test_layer_creation() {
            // Test creating a single layer
            let layer = Layer::new("test_layer".to_string(), 100, 10, false);
            
            // Check layer properties
            assert_eq!(layer.id, "test_layer");
            assert_eq!(layer.neurons, 10);
            assert_eq!(layer.is_output_layer, false);
            assert_eq!(layer.weights.len(), 10);
            assert_eq!(layer.biases.len(), 10);
        }
    }
    
    // Training tests
    mod training_tests {
        use crate::ai::models::advanced::NeuralNetwork;
        use crate::ai::engine::tensor::Tensor;
        use crate::ai::training::trainers;
        
        #[test]
        fn test_basic_training() {
            // Create test data
            let input_data = vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.7, 0.8, 0.9],
            ];
            
            let labels = vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ];
            
            // Create neural network
            let mut model = NeuralNetwork::new("test_training_model", vec![3, 5, 2]);
            
            // Training parameters
            let epochs = 5;
            let batch_size = 1;
            let learning_rate = 0.01;
            
            // Convert to tensors
            let input_tensor = Tensor::new(input_data.clone(), vec![3, 3]);
            let label_tensor = Tensor::new(labels.clone(), vec![3, 2]);
            
            // Train model
            let result = trainers::train_model(&mut model, &input_tensor, &label_tensor, epochs, batch_size, learning_rate);
            
            assert!(result.is_ok());
        }
    }
    
    // Integration tests
    mod integration_tests {
        use crate::integrations;
        
        #[test]
        fn test_tensorflow_compatibility() {
            // Check if TensorFlow integration can be initialized
            let result = integrations::tensorflow::initialize();
            assert!(result.is_ok());
        }
        
        #[test]
        fn test_pytorch_compatibility() {
            // Check if PyTorch integration can be initialized
            let result = integrations::pytorch::initialize();
            assert!(result.is_ok());
        }
        
        #[test]
        fn test_huggingface_integration() {
            // Check if HuggingFace integration can be initialized
            let result = integrations::huggingface::initialize();
            assert!(result.is_ok());
        }
        
        #[test]
        fn test_web_integration() {
            // Check if web integration can be initialized
            let result = integrations::web::initialize();
            assert!(result.is_ok());
        }
    }
    
    // Export tests
    mod export_tests {
        use crate::ai::models::advanced::NeuralNetwork;
        use crate::ai::deployment::exporters;
        
        #[test]
        fn test_export_model() {
            // Create a model to export
            let model = NeuralNetwork::new("export_test_model", vec![10, 5, 2]);
            
            // Export to ONNX format
            let result = exporters::export_model(&model, "onnx", "test_export.onnx");
            assert!(result.is_ok());
        }
    }
}