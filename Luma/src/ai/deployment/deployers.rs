use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use serde::{Serialize, Deserialize};
use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::models::layers::Layer;
use crate::ai::engine::tensor::Tensor;

/// Model metadata for serialization
#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub architecture: Vec<usize>,
    pub version: String,
    pub creation_date: String,
    pub description: Option<String>,
}

/// Layer data for serialization
#[derive(Serialize, Deserialize)]
pub struct SerializedLayer {
    pub id: String,
    pub input_size: usize,
    pub output_size: usize,
    pub is_output_layer: bool,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

/// Complete model for serialization
#[derive(Serialize, Deserialize)]
pub struct SerializedModel {
    pub metadata: ModelMetadata,
    pub layers: Vec<SerializedLayer>,
}

/// Save model to disk in Luma format
pub fn save_model(model: &NeuralNetwork, path: &str, description: Option<&str>) -> Result<(), String> {
    // Create model metadata
    let current_date = chrono::Local::now().to_rfc3339();
    
    let mut architecture = Vec::new();
    if !model.layers.is_empty() {
        // First layer input size
        let first_layer = &model.layers[0];
        architecture.push(first_layer.weights[0].get_data().len());
        
        // Output size of each layer
        for layer in &model.layers {
            architecture.push(layer.neurons);
        }
    }
    
    let metadata = ModelMetadata {
        id: model.id.clone(),
        architecture,
        version: "1.0".to_string(),
        creation_date: current_date,
        description: description.map(|s| s.to_string()),
    };
    
    // Serialize layers
    let mut serialized_layers = Vec::new();
    
    for layer in &model.layers {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        // Extract weights and biases
        for i in 0..layer.neurons {
            weights.push(layer.weights[i].get_data().to_vec());
            biases.push(layer.biases[i].get_data()[0]);  // Assuming bias is a scalar
        }
        
        let serialized_layer = SerializedLayer {
            id: layer.id.clone(),
            input_size: if !weights.is_empty() { weights[0].len() } else { 0 },
            output_size: layer.neurons,
            is_output_layer: layer.is_output_layer,
            weights,
            biases,
        };
        
        serialized_layers.push(serialized_layer);
    }
    
    // Create complete serialized model
    let serialized_model = SerializedModel {
        metadata,
        layers: serialized_layers,
    };
    
    // Serialize to JSON
    let json = serde_json::to_string_pretty(&serialized_model)
        .map_err(|e| format!("Failed to serialize model: {}", e))?;
    
    // Write to file
    let mut file = File::create(path)
        .map_err(|e| format!("Failed to create file {}: {}", path, e))?;
    
    file.write_all(json.as_bytes())
        .map_err(|e| format!("Failed to write to file {}: {}", path, e))?;
    
    println!("Model saved successfully to {}", path);
    Ok(())
}

/// Load model from disk
pub fn load_model(path: &str) -> Result<NeuralNetwork, String> {
    // Check if file exists
    if !Path::new(path).exists() {
        return Err(format!("Model file {} does not exist", path));
    }
    
    // Read file
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open file {}: {}", path, e))?;
    
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read file {}: {}", path, e))?;
    
    // Parse JSON
    let serialized_model: SerializedModel = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
    
    // Recreate the model
    let mut layers = Vec::new();
    
    for layer_data in &serialized_model.layers {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        // Recreate weights and biases
        for i in 0..layer_data.output_size {
            if i < layer_data.weights.len() {
                let w = Tensor::with_grad(layer_data.weights[i].clone(), vec![layer_data.weights[i].len()]);
                weights.push(w);
            }
            
            if i < layer_data.biases.len() {
                let b = Tensor::with_grad(vec![layer_data.biases[i]], vec![1]);
                biases.push(b);
            }
        }
        
        // Create layer
        let mut layer = Layer::new(
            layer_data.id.clone(),
            layer_data.input_size,
            layer_data.output_size,
            layer_data.is_output_layer,
        );
        
        // Replace weights and biases with loaded ones
        layer.weights = weights;
        layer.biases = biases;
        
        layers.push(layer);
    }
    
    // Create and return the neural network
    let model = NeuralNetwork {
        id: serialized_model.metadata.id,
        layers,
    };
    
    println!("Model loaded successfully from {}", path);
    Ok(model)
}

/// Deploy model to a location (copying the model file)
pub fn deploy_model(model_path: &str, target: &str) -> Result<(), String> {
    // Check if source file exists
    if !Path::new(model_path).exists() {
        return Err(format!("Source model file {} does not exist", model_path));
    }
    
    // Read source file
    let mut source_file = File::open(model_path)
        .map_err(|e| format!("Failed to open source file {}: {}", model_path, e))?;
    
    let mut contents = Vec::new();
    source_file.read_to_end(&mut contents)
        .map_err(|e| format!("Failed to read source file {}: {}", model_path, e))?;
    
    // Write to target location
    let mut target_file = File::create(target)
        .map_err(|e| format!("Failed to create target file {}: {}", target, e))?;
    
    target_file.write_all(&contents)
        .map_err(|e| format!("Failed to write to target file {}: {}", target, e))?;
    
    println!("Model deployed successfully from {} to {}", model_path, target);
    Ok(())
}