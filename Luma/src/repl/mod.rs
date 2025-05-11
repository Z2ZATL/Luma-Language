use std::io::{self, Write};
use crate::ai::data::{loaders, augmentations, preprocessors, multi_modal};
use crate::ai::data::preprocessors::PreprocessingMethod;
use crate::ai::data::augmentations::AugmentationMethod;
use crate::ai::training::trainers::Trainer;
use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::models::optimizers::SGD;
use crate::ai::training::schedulers::LearningRateScheduler;
use crate::ai::training::callbacks::{LoggingCallback, CallbackList};
use crate::ai::evaluation::evaluators;
use crate::ai::deployment::{deployers, exporters};

// ฟังก์ชันช่วยในการแบ่ง parameters จากคำสั่ง
fn parse_parameters(input: &str) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();
    let parts: Vec<&str> = input.split_whitespace().collect();
    
    for part in parts {
        if part.contains('=') {
            let kv: Vec<&str> = part.splitn(2, '=').collect();
            if kv.len() == 2 {
                params.insert(kv[0].to_string(), kv[1].to_string());
            }
        }
    }
    
    params
}

// ฟังก์ชันสำหรับแสดงความช่วยเหลือ
fn show_help() {
    println!("Available commands:");
    println!("  load dataset \"path\" as dataset_name [lazy=true|false]");
    println!("  load multimodal \"path\" as dataset_name");
    println!("  print dataset dataset_name");
    println!("  split dataset dataset_name ratio=0.3");
    println!("  preprocess dataset_name method=normalize|scale|log|sqrt as new_name");
    println!("  augment dataset_name method=noise(0.1)|dropout(0.2)|rotation|mirror|shuffle as new_name");
    println!("  rename dataset source_name as target_name");
    println!("  list datasets");
    println!("  clear datasets");
    println!("  train epochs=10 batch_size=32 learning_rate=0.01");
    println!("  evaluate model dataset_name"); 
    println!("  save model \"path/to/file.luma\"");
    println!("  load_model \"path/to/file.luma\"");
    println!("  export model format=\"onnx|tensorflow|wasm|json\" path=\"output_path\"");
    println!("  exit");
    println!("  help");
}

pub fn start_repl() {
    println!("Luma REPL v1.0.0 (type 'help' for commands, 'exit' to quit)");
    loop {
        print!("Luma> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" {
            println!("Exiting Luma REPL...");
            break;
        }

        if input == "help" {
            show_help();
            continue;
        }

        // แยกคำสั่งด้วย space แต่รักษา quoted strings
        let mut parts = Vec::new();
        let mut current_part = String::new();
        let mut in_quotes = false;
        
        for c in input.chars() {
            if c == '"' {
                in_quotes = !in_quotes;
                current_part.push(c);
            } else if c == ' ' && !in_quotes {
                if !current_part.is_empty() {
                    parts.push(current_part);
                    current_part = String::new();
                }
            } else {
                current_part.push(c);
            }
        }
        
        if !current_part.is_empty() {
            parts.push(current_part);
        }
        
        if parts.is_empty() {
            continue;
        }

        // Debug: แสดงคำสั่งที่รับมา
        println!("Debug: Received {} parts in command", parts.len());
        for (i, p) in parts.iter().enumerate() {
            println!("Debug: Part {}: '{}'", i, p);
        }

        let command = parts[0].as_str();
        match command {
            "load" => {
                if parts.len() >= 4 && (parts[1] == "dataset" || parts[1] == "multimodal") && parts.len() >= 5 && parts[3] == "as" {
                    let data_type = parts[1].as_str();
                    let path = parts[2].trim_matches('"');
                    let name = parts[4].as_str();
                    
                    // Parse additional parameters
                    let params = parse_parameters(&parts[5..].join(" "));
                    let lazy = params.get("lazy").map_or(false, |v| v == "true");
                    
                    if data_type == "dataset" {
                        match loaders::load_dataset(path, name, lazy, None) {
                            Ok(_) => println!("Dataset {} loaded successfully", name),
                            Err(e) => println!("Error loading dataset: {}", e),
                        }
                    } else if data_type == "multimodal" {
                        match multi_modal::load_dataset(path, name) {
                            Ok(_) => println!("Multi-modal dataset {} loaded successfully", name),
                            Err(e) => println!("Error loading multi-modal dataset: {}", e),
                        }
                    }
                } else {
                    println!("Usage:");
                    println!("  load dataset \"path\" as name [lazy=true|false]");
                    println!("  load multimodal \"path\" as name");
                }
            },
            "print" => {
                if parts.len() >= 3 && parts[1] == "dataset" {
                    let name = parts[2].trim_matches('"');
                    loaders::print_dataset(name);
                } else {
                    println!("Usage: print dataset dataset_name");
                }
            },
            "list" => {
                if parts.len() >= 2 && parts[1] == "datasets" {
                    loaders::list_datasets();
                } else {
                    println!("Usage: list datasets");
                }
            },
            "rename" => {
                if parts.len() >= 5 && parts[1] == "dataset" && parts[3] == "as" {
                    let source_name = parts[2].trim_matches('"');
                    let target_name = parts[4].trim_matches('"');
                    
                    // Check if source dataset exists
                    if let Some(source_dataset) = loaders::get_dataset(source_name) {
                        // Make a copy of all dataset properties
                        let source_data = source_dataset.get_data().clone();
                        let source_labels = source_dataset.get_labels().clone();
                        let headers = source_dataset.get_headers().clone().unwrap_or_else(Vec::new);
                        
                        // Create a new dataset with the target name
                        match loaders::load_dataset_from_memory(target_name, &source_data, &source_labels, &headers) {
                            Ok(_) => {
                                println!("Renamed dataset '{}' to '{}'", source_name, target_name);
                                // Keep the original dataset (don't delete)
                            },
                            Err(e) => println!("Error renaming dataset: {}", e),
                        }
                    } else {
                        println!("Source dataset '{}' not found", source_name);
                    }
                } else {
                    println!("Usage: rename dataset source_name as target_name");
                }
            },
            "clear" => {
                if parts.len() >= 2 && parts[1] == "datasets" {
                    loaders::clear_datasets();
                    println!("All datasets cleared");
                } else {
                    println!("Usage: clear datasets");
                }
            },
            "split" => {
                if parts.len() >= 4 && parts[1] == "dataset" {
                    let name = parts[2].trim_matches('"');
                    let params = parse_parameters(&parts[3..].join(" "));
                    
                    if let Some(ratio_str) = params.get("ratio") {
                        if let Ok(ratio) = ratio_str.parse::<f64>() {
                            match loaders::split_dataset(name, ratio) {
                                Ok((train, test)) => println!("Split dataset into '{}' and '{}'", train, test),
                                Err(e) => println!("Error splitting dataset: {}", e),
                            }
                        } else {
                            println!("Invalid ratio: {}", ratio_str);
                        }
                    } else {
                        println!("Usage: split dataset dataset_name ratio=0.3");
                    }
                } else {
                    println!("Usage: split dataset dataset_name ratio=0.3");
                }
            },
            "preprocess" => {
                // 1. ตรวจสอบรูปแบบคำสั่ง "preprocess dataset_name method=normalize as new_name"
                println!("Debug: Preprocessing with {} parts", parts.len());
                if parts.len() == 5 && parts[3] == "as" {
                    // 2. ดึงข้อมูลจากคำสั่ง
                    let dataset_name = parts[1].trim_matches('"');
                    let method_part = &parts[2];
                    let output_name = parts[4].trim_matches('"');
                    
                    println!("Debug: Preprocessing dataset='{}', method='{}', output='{}'", 
                             dataset_name, method_part, output_name);
                    
                    // 3. ตรวจสอบว่า method มีรูปแบบถูกต้อง
                    if method_part.starts_with("method=") {
                        // 4. ดึงรายละเอียดวิธีการ
                        let method_name = &method_part["method=".len()..];
                        println!("Debug: Using method '{}'", method_name);
                        
                        // 5. แปลงเป็น enum PreprocessingMethod ตามที่ระบุ
                        let method = match method_name {
                            "normalize" => PreprocessingMethod::Normalize,
                            "scale" => PreprocessingMethod::MinMaxScale,
                            "log" => PreprocessingMethod::LogTransform,
                            "sqrt" => PreprocessingMethod::SqrtTransform,
                            _ => {
                                println!("Unknown preprocessing method: {}", method_name);
                                continue;
                            }
                        };
                        
                        // 6. เรียกฟังก์ชัน preprocess_dataset
                        println!("Debug: Calling preprocess_dataset with {} and output {}", dataset_name, output_name);
                        match preprocessors::preprocess_dataset(dataset_name, method, Some(output_name)) {
                            Ok(name) => println!("Created preprocessed dataset '{}'", name),
                            Err(e) => println!("Error preprocessing dataset: {}", e),
                        }
                    } else {
                        println!("Usage: preprocess dataset_name method=normalize|scale|log|sqrt as new_name");
                    }
                } else {
                    println!("Usage: preprocess dataset_name method=normalize|scale|log|sqrt as new_name");
                }
            },
            "augment" => {
                // 1. ตรวจสอบรูปแบบคำสั่ง "augment dataset_name method=... as new_name"
                println!("Debug: Augmenting with {} parts", parts.len());
                if parts.len() == 5 && parts[3] == "as" {
                    // 2. ดึงข้อมูลจากคำสั่ง
                    let dataset_name = parts[1].trim_matches('"');
                    let method_part = &parts[2];
                    let output_name = parts[4].trim_matches('"');
                    
                    println!("Debug: Augmenting dataset='{}', method='{}', output='{}'", 
                             dataset_name, method_part, output_name);
                    
                    // 3. ตรวจสอบว่า method มีรูปแบบถูกต้อง
                    if method_part.starts_with("method=") {
                        // 4. ดึงรายละเอียดวิธีการ
                        let method_str = &method_part["method=".len()..];
                        println!("Debug: Using method '{}'", method_str);
                        
                        // 5. แปลงเป็น enum AugmentationMethod ตามที่ระบุ
                        let method = if method_str.starts_with("noise(") && method_str.ends_with(")") {
                            let param_str = &method_str[6..method_str.len()-1];
                            println!("Debug: Noise parameter: {}", param_str);
                            if let Ok(amount) = param_str.parse::<f64>() {
                                AugmentationMethod::Noise(amount)
                            } else {
                                println!("Invalid noise parameter: {}", param_str);
                                continue;
                            }
                        } else if method_str.starts_with("dropout(") && method_str.ends_with(")") {
                            let param_str = &method_str[8..method_str.len()-1];
                            println!("Debug: Dropout parameter: {}", param_str);
                            if let Ok(prob) = param_str.parse::<f64>() {
                                AugmentationMethod::Dropout(prob)
                            } else {
                                println!("Invalid dropout parameter: {}", param_str);
                                continue;
                            }
                        } else if method_str == "rotation" {
                            AugmentationMethod::Rotation
                        } else if method_str == "mirror" {
                            AugmentationMethod::Mirror
                        } else if method_str == "shuffle" {
                            AugmentationMethod::Shuffle
                        } else {
                            println!("Unknown augmentation method: {}", method_str);
                            continue;
                        };
                        
                        // 6. เรียกฟังก์ชัน augment_dataset
                        println!("Debug: Calling augment_dataset with {} and output {}", dataset_name, output_name);
                        match augmentations::augment_dataset(dataset_name, output_name, method) {
                            Ok(_) => println!("Created augmented dataset '{}'", output_name),
                            Err(e) => println!("Error augmenting dataset: {}", e),
                        }
                    } else {
                        println!("Usage: augment dataset_name method=noise(0.1)|dropout(0.2)|rotation|mirror|shuffle as new_name");
                    }
                } else {
                    println!("Usage: augment dataset_name method=noise(0.1)|dropout(0.2)|rotation|mirror|shuffle as new_name");
                }
            },
            "train" => {
                println!("Debug: Received {} parts in command", parts.len());
                for (i, p) in parts.iter().enumerate() {
                    println!("Debug: Part {}: '{}'", i, p);
                }
                
                let params = parse_parameters(&parts[1..].join(" "));
                
                let epochs = params.get("epochs").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                let batch_size = params.get("batch_size").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                let learning_rate = params.get("learning_rate").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                
                if epochs > 0 && batch_size > 0 && learning_rate > 0.0 {
                    if let Some(dataset) = loaders::get_dataset("training_data") {
                        // Determine input and output dimensions based on dataset
                        let input_dim = dataset.get_feature_count();
                        let output_dim = 1; // Default to 1 for binary classification
                        
                        // Create a network with appropriate layer sizes for this dataset
                        // with hidden layers that taper down in size
                        let layer_sizes = vec![input_dim, 
                                              (input_dim * 2).max(8), // First hidden layer
                                              (input_dim).max(4),     // Second hidden layer 
                                              output_dim];
                              
                        println!("Creating neural network with layer sizes: {:?}", layer_sizes);
                        let model = NeuralNetwork::new("nn_model", layer_sizes);
                        let optimizer = SGD::new(0.9, &model.layers);
                        let scheduler = LearningRateScheduler::new(0.1); // Slightly higher start rate
                        
                        let mut trainer = Trainer::new(model, optimizer, scheduler);
                        let mut callbacks = CallbackList::new();
                        callbacks.add(LoggingCallback::new(true));
                        trainer.add_callback(Box::new(callbacks));
                        
                        // Get training labels - each row should have at least one label
                        let labels = dataset.get_labels();
                        
                        if !labels.is_empty() {
                            println!("Training model with:");
                            println!("  - {} epochs", epochs);
                            println!("  - batch size {}", batch_size);
                            println!("  - learning rate {}", learning_rate);
                            println!("  - {} training examples", dataset.get_data().len());
                            println!("  - {} features per example", input_dim);
                            
                            // Set debug level for training - 1 is a good balance
                            trainer.set_debug_level(1);
                            
                            // Start training with the given parameters
                            trainer.train(&dataset, &labels, epochs, batch_size, learning_rate);
                            println!("Training completed!");
                        } else {
                            println!("Error: No labels found in dataset. Make sure your dataset has label columns.");
                        }
                    } else {
                        println!("Dataset 'training_data' not found. Load a dataset first and rename it to 'training_data'");
                        println!("Tip: Try 'load dataset \"path\" as training_data'");
                        println!("Or rename an existing dataset: rename dataset existing_name as training_data");
                    }
                } else {
                    println!("Usage: train epochs=10 batch_size=32 learning_rate=0.01");
                    println!("All parameters must be positive values.");
                }
            },
            "evaluate" => {
                // Check if we have enough parts like "evaluate model dataset_name"
                if parts.len() >= 3 && parts[1] == "model" {
                    let dataset_name = parts[2].trim_matches('"');
                    
                    // Get the dataset for evaluation
                    if let Some(dataset) = loaders::get_dataset(dataset_name) {
                        println!("Found evaluation dataset '{}'", dataset_name);
                        
                        // Get the current model
                        // In a real implementation, we would have a way to access the trained model
                        // For now, we'll use a placeholder test model
                        let input_dim = if !dataset.get_data().is_empty() {
                            dataset.get_data()[0].len()
                        } else {
                            println!("Error: Evaluation data is empty");
                            continue;
                        };
                        
                        let output_dim = 1; // Binary classification
                        let mut model = NeuralNetwork::new(
                            "test_model",
                            vec![input_dim, 5, output_dim],
                        );
                        
                        // Evaluate the model on the dataset
                        let results = evaluators::evaluate_trained_model(&mut model, &dataset);
                        
                        // Display the evaluation results
                        evaluators::print_evaluation_results(&results);
                    } else {
                        println!("Dataset '{}' not found. Load a dataset first.", dataset_name);
                    }
                } else {
                    println!("Usage: evaluate model dataset_name");
                }
            },
            "save" => {
                // Check if we have enough parts like "save model path/to/file.luma"
                if parts.len() >= 3 && parts[1] == "model" {
                    let file_path = parts[2].trim_matches('"');
                    
                    // In a real implementation, we would have a way to access the trained model
                    // For now, we'll use a placeholder test model
                    let model = NeuralNetwork::new(
                        "test_model",
                        vec![10, 5, 1],
                    );
                    
                    // Save the model to the specified file
                    match deployers::save_model(&model, file_path, None) {
                        Ok(()) => println!("Model saved successfully to {}", file_path),
                        Err(e) => println!("Error saving model: {}", e),
                    }
                } else {
                    println!("Usage: save model \"path/to/file.luma\"");
                }
            },
            "load_model" => {
                // Check if we have enough parts like "load_model path/to/file.luma"
                if parts.len() >= 2 {
                    let file_path = parts[1].trim_matches('"');
                    
                    // Load the model from the specified file
                    match deployers::load_model(file_path) {
                        Ok(model) => {
                            println!("Model loaded successfully from {}", file_path);
                            println!("Model ID: {}", model.id);
                            println!("Number of layers: {}", model.layers.len());
                            
                            // In a real implementation, we would store this model for later use
                        },
                        Err(e) => println!("Error loading model: {}", e),
                    }
                } else {
                    println!("Usage: load_model \"path/to/file.luma\"");
                }
            },
            "export" => {
                // Check if we have the right format like "export model format="onnx" path="output_path"
                if parts.len() >= 2 && parts[1] == "model" {
                    let params = parse_parameters(&parts.join(" "));
                    
                    if let (Some(format), Some(path)) = (params.get("format"), params.get("path")) {
                        let format = format.trim_matches('"');
                        let path = path.trim_matches('"');
                        
                        // In a real implementation, we would have a way to access the trained model
                        // For now, we'll use a placeholder test model
                        let model = NeuralNetwork::new(
                            "test_model",
                            vec![10, 5, 1],
                        );
                        
                        // Export the model to the specified format and path
                        match exporters::export_model(&model, format, path) {
                            Ok(output_path) => println!("Model exported successfully to {} format at {}", format, output_path),
                            Err(e) => println!("Error exporting model: {}", e),
                        }
                    } else {
                        println!("Usage: export model format=\"onnx|tensorflow|wasm|json\" path=\"output_path\"");
                    }
                } else {
                    println!("Usage: export model format=\"onnx|tensorflow|wasm|json\" path=\"output_path\"");
                }
            },
            _ => println!("Unknown command: '{}'. Type 'help' for available commands.", command),
        }
        io::stdout().flush().unwrap();
    }
}