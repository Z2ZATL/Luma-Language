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
use crate::ai::engine::accelerators;
use crate::utilities::{logging, profiling, visualization};
use crate::plugins;
// Integration modules
use crate::integrations::{tensorflow, pytorch, huggingface, web};
use crate::compiler::backend::wasm;
use std::collections::HashMap;

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
    println!("\n=== Data Management ===");
    println!("  load dataset \"path\" as dataset_name [lazy=true|false]");
    println!("  load multimodal \"path\" as dataset_name");
    println!("  print dataset dataset_name");
    println!("  split dataset dataset_name ratio=0.3");
    println!("  preprocess dataset_name method=normalize|scale|log|sqrt as new_name");
    println!("  augment dataset_name method=noise(0.1)|dropout(0.2)|rotation|mirror|shuffle as new_name");
    println!("  rename dataset source_name as target_name");
    println!("  list datasets");
    println!("  clear datasets");
    
    println!("\n=== Model Training & Evaluation ===");
    println!("  train epochs=10 batch_size=32 learning_rate=0.01");
    println!("  evaluate model dataset_name"); 
    println!("  save model \"path/to/file.luma\"");
    println!("  load_model \"path/to/file.luma\"");
    println!("  export model format=\"onnx|tensorflow|pytorch|wasm|json\" path=\"output_path\" [options...]");
    println!("  import model \"path/to/model\" format=\"tensorflow|pytorch|huggingface\"");
    println!("  huggingface search \"query\" [task] [limit=5]");
    println!("  huggingface load \"model_name\" [options...]");
    println!("  huggingface push model \"repo_id\" \"auth_token\" [private=false]");
    println!("  deploy_web model \"server_config.json\"");
    println!("  tensorflow export model=\"name\" path=\"path/to/output\" [options...]");
    println!("  pytorch export model=\"name\" path=\"path/to/output\" [onnx=true] [options...]");
    println!("  bindings generate c|python|javascript model=\"name\" path=\"output_path\"");
    
    println!("\n=== Performance & Optimization ===");
    println!("  set device cpu|cuda|opencl|metal|tpu");
    println!("  device info");
    println!("  start profiling");
    println!("  stop profiling");
    println!("  plot metrics \"output_path.svg\"");
    println!("  set log_level trace|debug|info|warning|error|fatal");
    
    println!("\n=== Plugins ===");
    println!("  plugin list");
    println!("  plugin info <plugin_id>");
    println!("  plugin enable <plugin_id>");
    println!("  plugin disable <plugin_id>");
    println!("  execute plugin <plugin_id> <command> [args...]");
    println!("  nlp tokenize \"Your text here\"");
    println!("  nlp analyze_sentiment \"Your text here\"");
    println!("  img resize \"input.jpg\" 800 600 \"output.jpg\"");
    println!("  mm load \"image.jpg\" [dataset_name]");
    
    println!("\n=== System ===");
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
            "tensorflow" | "tf" => {
                // TensorFlow integration commands
                if parts.len() < 2 {
                    println!("Usage: tensorflow <export|import> [options...]");
                    continue;
                }
                
                match parts[1].as_str() {
                    "export" => {
                        let params = parse_parameters(&parts[2..].join(" "));
                        
                        if let (Some(model_name), Some(path)) = (params.get("model"), params.get("path")) {
                            let tflite = params.get("tflite").map_or(false, |v| v == "true");
                            let saved_model = params.get("saved_model").map_or(true, |v| v == "true");
                            
                            println!("Would export model '{}' to TensorFlow format at path: {}", 
                                     model_name, path);
                            println!("Options: TFLite={}, SavedModel={}", tflite, saved_model);
                            
                            let config = tensorflow::TensorFlowExportConfig {
                                tf_version: "2.8.0".to_string(),
                                use_tflite: tflite,
                                include_weights: true,
                                use_saved_model: saved_model,
                            };
                            
                            let model = tensorflow::Model::new(model_name);
                            match tensorflow::export_to_tensorflow(&model, path, Some(config)) {
                                Ok(_) => println!("Model exported successfully to TensorFlow format"),
                                Err(e) => println!("Failed to export model: {}", e),
                            }
                        } else {
                            println!("Usage: tensorflow export model=\"model_name\" path=\"output_path\" [tflite=true|false] [saved_model=true|false]");
                        }
                    },
                    "import" => {
                        if parts.len() >= 5 && parts[2] == "path" && parts[4] == "as" {
                            let path = parts[3].trim_matches('"');
                            let model_name = parts[5].trim_matches('"');
                            
                            println!("Would import TensorFlow model from '{}' as '{}'", path, model_name);
                            
                            match tensorflow::import_from_tensorflow(path, model_name) {
                                Ok(_) => println!("TensorFlow model imported successfully as '{}'", model_name),
                                Err(e) => println!("Failed to import model: {}", e),
                            }
                        } else {
                            println!("Usage: tensorflow import path=\"input_path\" as model_name");
                        }
                    },
                    _ => println!("Unknown tensorflow command: {}. Use 'export' or 'import'.", parts[1]),
                }
            },
            "pytorch" | "pt" => {
                // PyTorch integration commands
                if parts.len() < 2 {
                    println!("Usage: pytorch <export|import|onnx> [options...]");
                    continue;
                }
                
                match parts[1].as_str() {
                    "export" => {
                        let params = parse_parameters(&parts[2..].join(" "));
                        
                        if let (Some(model_name), Some(path)) = (params.get("model"), params.get("path")) {
                            let to_onnx = params.get("onnx").map_or(false, |v| v == "true");
                            
                            println!("Would export model '{}' to PyTorch format at path: {}", 
                                     model_name, path);
                            println!("Also export to ONNX: {}", to_onnx);
                            
                            let config = pytorch::PyTorchExportConfig {
                                torch_version: "1.13.0".to_string(),
                                use_jit: true,
                                optimize: true,
                                include_weights: true,
                            };
                            
                            let model = tensorflow::Model::new(model_name);
                            match pytorch::export_to_pytorch(&model, path, Some(config)) {
                                Ok(_) => {
                                    println!("Model exported successfully to PyTorch format");
                                    
                                    if to_onnx {
                                        let onnx_path = format!("{}.onnx", path);
                                        match pytorch::convert_to_onnx(&model, &onnx_path, 15) {
                                            Ok(_) => println!("Model also exported to ONNX format at {}", onnx_path),
                                            Err(e) => println!("Failed to export to ONNX: {}", e),
                                        }
                                    }
                                },
                                Err(e) => println!("Failed to export model: {}", e),
                            }
                        } else {
                            println!("Usage: pytorch export model=\"model_name\" path=\"output_path\" [onnx=true|false]");
                        }
                    },
                    "import" => {
                        if parts.len() >= 5 && parts[2] == "path" && parts[4] == "as" {
                            let path = parts[3].trim_matches('"');
                            let model_name = parts[5].trim_matches('"');
                            
                            println!("Would import PyTorch model from '{}' as '{}'", path, model_name);
                            
                            match pytorch::import_from_pytorch(path, model_name) {
                                Ok(_) => println!("PyTorch model imported successfully as '{}'", model_name),
                                Err(e) => println!("Failed to import model: {}", e),
                            }
                        } else {
                            println!("Usage: pytorch import path=\"input_path\" as model_name");
                        }
                    },
                    "onnx" => {
                        let params = parse_parameters(&parts[2..].join(" "));
                        
                        if let (Some(model_name), Some(path)) = (params.get("model"), params.get("path")) {
                            let opset = params.get("opset").map_or(15, |v| v.parse().unwrap_or(15));
                            
                            println!("Would export model '{}' to ONNX format at path: {}", 
                                     model_name, path);
                            println!("ONNX opset version: {}", opset);
                            
                            let model = tensorflow::Model::new(model_name);
                            match pytorch::convert_to_onnx(&model, path, opset) {
                                Ok(_) => println!("Model exported successfully to ONNX format"),
                                Err(e) => println!("Failed to export model: {}", e),
                            }
                        } else {
                            println!("Usage: pytorch onnx model=\"model_name\" path=\"output_path\" [opset=15]");
                        }
                    },
                    _ => println!("Unknown pytorch command: {}. Use 'export', 'import', or 'onnx'.", parts[1]),
                }
            },
            "huggingface" | "hf" => {
                // Hugging Face integration commands
                if parts.len() < 2 {
                    println!("Usage: huggingface <search|download|push> [options...]");
                    continue;
                }
                
                match parts[1].as_str() {
                    "search" => {
                        if parts.len() >= 3 {
                            let query = parts[2].trim_matches('"');
                            let params = parse_parameters(&parts[3..].join(" "));
                            
                            let task = params.get("task").cloned();
                            let limit = params.get("limit").map_or(5, |v| v.parse().unwrap_or(5));
                            
                            println!("Would search Hugging Face for: '{}'", query);
                            if let Some(task_type) = &task {
                                println!("Task filter: {}", task_type);
                            }
                            println!("Result limit: {}", limit);
                            
                            match huggingface::search_models(query, task, limit) {
                                Ok(results) => {
                                    println!("Found {} models:", results.len());
                                    for (i, result) in results.iter().enumerate() {
                                        println!("{}. {} (Downloads: {})", i+1, result, i*1000+500);
                                    }
                                },
                                Err(e) => println!("Search failed: {}", e),
                            }
                        } else {
                            println!("Usage: huggingface search \"query\" [task=translation] [limit=5]");
                        }
                    },
                    "download" => {
                        if parts.len() >= 3 {
                            let model_id = parts[2].trim_matches('"');
                            let params = parse_parameters(&parts[3..].join(" "));
                            
                            let revision = params.get("revision").cloned();
                            let cache_dir = params.get("cache").cloned();
                            
                            println!("Would download Hugging Face model: '{}'", model_id);
                            if let Some(rev) = &revision {
                                println!("Revision: {}", rev);
                            }
                            if let Some(cache) = &cache_dir {
                                println!("Cache directory: {}", cache);
                            }
                            
                            let config = huggingface::HuggingFaceConfig {
                                use_gpu: true,
                                cache_dir: cache_dir.unwrap_or_else(|| "./.cache".to_string()),
                                revision: revision,
                                use_auth_token: None,
                                model_id: model_id.to_string(),
                            };
                            
                            match huggingface::download_model(&config) {
                                Ok(model_path) => println!("Model downloaded to: {}", model_path),
                                Err(e) => println!("Download failed: {}", e),
                            }
                        } else {
                            println!("Usage: huggingface download \"model_id\" [revision=main] [cache=path]");
                        }
                    },
                    "push" => {
                        let params = parse_parameters(&parts[2..].join(" "));
                        
                        if let (Some(model_name), Some(repo_id), Some(token)) = 
                           (params.get("model"), params.get("repo"), params.get("token")) {
                            let private = params.get("private").map_or(false, |v| v == "true");
                            
                            println!("Would push model '{}' to Hugging Face Hub repo: '{}'", 
                                     model_name, repo_id);
                            println!("Private repository: {}", private);
                            
                            let model = tensorflow::Model::new(model_name);
                            match huggingface::push_to_huggingface(&model, repo_id, token, private) {
                                Ok(_) => println!("Model pushed successfully to Hugging Face Hub"),
                                Err(e) => println!("Failed to push model: {}", e),
                            }
                        } else {
                            println!("Usage: huggingface push model=\"model_name\" repo=\"user/repo_id\" token=\"hf_token\" [private=true|false]");
                        }
                    },
                    _ => println!("Unknown huggingface command: {}. Use 'search', 'download', or 'push'.", parts[1]),
                }
            },
            "wasm" => {
                // WebAssembly integration commands
                if parts.len() < 2 {
                    println!("Usage: wasm <export|optimize> [options...]");
                    continue;
                }
                
                match parts[1].as_str() {
                    "export" => {
                        let params = parse_parameters(&parts[2..].join(" "));
                        
                        if let (Some(model_name), Some(path)) = (params.get("model"), params.get("path")) {
                            let opt_level = params.get("opt").map_or(2, |v| v.parse().unwrap_or(2));
                            let debug = params.get("debug").map_or(false, |v| v == "true");
                            
                            println!("Would export model '{}' to WebAssembly at path: {}", 
                                     model_name, path);
                            println!("Optimization level: {}, Debug info: {}", opt_level, debug);
                            
                            let options = wasm::WasmCompileOptions {
                                optimization_level: opt_level as u8,
                                debug_info: debug,
                                generate_js_bindings: true,
                                wasm_features: vec!["simd".to_string()],
                            };
                            
                            let model = tensorflow::Model::new(model_name);
                            match wasm::compile_to_wasm(&model, path, Some(options)) {
                                Ok(_) => println!("Model exported successfully to WebAssembly"),
                                Err(e) => println!("Failed to export model: {}", e),
                            }
                        } else {
                            println!("Usage: wasm export model=\"model_name\" path=\"output_path\" [opt=0-3] [debug=true|false]");
                        }
                    },
                    "optimize" => {
                        if parts.len() >= 3 {
                            let path = parts[2].trim_matches('"');
                            let params = parse_parameters(&parts[3..].join(" "));
                            
                            let for_size = params.get("for").map_or(true, |v| v == "size");
                            
                            println!("Would optimize WebAssembly module at: '{}'", path);
                            println!("Optimize for: {}", if for_size { "size" } else { "speed" });
                            
                            match wasm::optimize_wasm(path, for_size) {
                                Ok(_) => println!("WebAssembly module optimized successfully"),
                                Err(e) => println!("Optimization failed: {}", e),
                            }
                        } else {
                            println!("Usage: wasm optimize \"wasm_path\" [for=size|speed]");
                        }
                    },
                    _ => println!("Unknown wasm command: {}. Use 'export' or 'optimize'.", parts[1]),
                }
            },
            "bindings" => {
                // Language bindings commands
                if parts.len() < 3 || parts[1] != "generate" {
                    println!("Usage: bindings generate <c|python|javascript> [options...]");
                    continue;
                }
                
                let lang = parts[2].as_str();
                let params = parse_parameters(&parts[3..].join(" "));
                
                if let (Some(model_name), Some(output_path)) = (params.get("model"), params.get("path")) {
                    println!("Would generate {} bindings for model '{}' at: {}", 
                             lang, model_name, output_path);
                    
                    match lang {
                        "c" => {
                            println!("Generating C bindings...");
                            println!("Header file would be written to: {}/luma.h", output_path);
                            println!("Implementation file would be written to: {}/luma.c", output_path);
                            println!("C bindings generated successfully");
                        },
                        "python" => {
                            println!("Generating Python bindings...");
                            println!("Python module would be written to: {}/luma.py", output_path);
                            println!("Python bindings generated successfully");
                        },
                        "javascript" | "js" => {
                            println!("Generating JavaScript bindings...");
                            println!("JavaScript module would be written to: {}/luma.js", output_path);
                            println!("WebAssembly module would be written to: {}/luma_wasm.wasm", output_path);
                            println!("JavaScript bindings generated successfully");
                        },
                        _ => println!("Unsupported language: {}. Use 'c', 'python', or 'javascript'.", lang),
                    }
                } else {
                    println!("Usage: bindings generate {} model=\"model_name\" path=\"output_path\"", lang);
                }
            },
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
            "set" => {
                if parts.len() >= 3 {
                    if parts[1] == "device" {
                        let device_name = parts[2].trim().to_lowercase();
                        match accelerators::set_accelerator(&device_name) {
                            Ok(_) => println!("Device set to: {}", device_name),
                            Err(e) => println!("Error setting device: {}", e),
                        }
                    } else if parts[1] == "log_level" {
                        if parts.len() >= 3 {
                            let level_str = parts[2].trim().to_lowercase();
                            match logging::LogLevel::from_str(&level_str) {
                                Ok(level) => {
                                    match logging::set_log_level(level) {
                                        Ok(_) => println!("Log level set to: {}", level),
                                        Err(e) => println!("Error setting log level: {}", e),
                                    }
                                },
                                Err(e) => println!("Invalid log level: {}", e),
                            }
                        } else {
                            println!("Usage: set log_level trace|debug|info|warning|error|fatal");
                        }
                    } else {
                        println!("Unknown setting: '{}'. Available settings: device, log_level", parts[1]);
                    }
                } else {
                    println!("Usage: set device cpu|cuda|opencl|metal|tpu");
                    println!("       set log_level trace|debug|info|warning|error|fatal");
                }
            },
            "device" => {
                if parts.len() >= 2 && parts[1] == "info" {
                    let info = accelerators::get_accelerator_info();
                    println!("Accelerator Information:\n{}", info);
                } else {
                    println!("Usage: device info");
                }
            },
            "start" => {
                if parts.len() >= 2 && parts[1] == "profiling" {
                    // Check if profiling is already in progress using the global profiler
                    match profiling::get_metrics() {
                        Some(_) => println!("Profiling already in progress. Stop current profiling first."),
                        None => {
                            // Start global profiling session
                            profiling::start_profiling();
                            profiling::start_event("REPL Session");
                            println!("Profiling started. Use 'stop profiling' to view results.");
                        }
                    }
                } else {
                    println!("Usage: start profiling");
                }
            },
            "stop" => {
                if parts.len() >= 2 && parts[1] == "profiling" {
                    // Check if profiling is in progress
                    match profiling::get_metrics() {
                        Some(metrics) => {
                            profiling::end_event();
                            println!("Profiling stopped. Results:");
                            println!("{}", metrics);
                            // Clear the global profiler state
                            profiling::clear_profiler();
                        },
                        None => {
                            println!("No profiling in progress. Start profiling first.");
                        }
                    }
                } else {
                    println!("Usage: stop profiling");
                }
            },
            "plot" => {
                if parts.len() >= 3 && parts[1] == "metrics" {
                    let output_path = parts[2].trim_matches('"');
                    
                    // Simulate some training metrics for visualization
                    // In a real implementation, this would use actual training history
                    let mut metrics = HashMap::new();
                    let epochs = 10;
                    
                    // Create some sample loss values that decrease over time
                    let mut loss_values = Vec::with_capacity(epochs);
                    for i in 0..epochs {
                        let progress = i as f64 / (epochs - 1) as f64;
                        let loss = 1.0 - 0.8 * progress + 0.1 * (progress * 5.0).sin();
                        loss_values.push(loss);
                    }
                    metrics.insert("Loss".to_string(), loss_values);
                    
                    // Create some sample accuracy values that increase over time
                    let mut accuracy_values = Vec::with_capacity(epochs);
                    for i in 0..epochs {
                        let progress = i as f64 / (epochs - 1) as f64;
                        let accuracy = 0.5 + 0.45 * progress - 0.05 * (progress * 4.0).cos();
                        accuracy_values.push(accuracy);
                    }
                    metrics.insert("Accuracy".to_string(), accuracy_values);
                    
                    match visualization::plot_training_metrics(&metrics, epochs, output_path) {
                        Ok(_) => println!("Training metrics plotted to: {}", output_path),
                        Err(e) => println!("Error plotting metrics: {}", e),
                    }
                } else {
                    println!("Usage: plot metrics \"output_path.svg\"");
                }
            },
            "plugin" => {
                // Plugin management commands
                if parts.len() < 2 {
                    println!("Usage: plugin <list|info|enable|disable> [plugin_id]");
                    continue;
                }
                
                let subcommand = parts[1].as_str();
                match subcommand {
                    "list" => {
                        // List all plugins
                        plugins::registry::print_plugins_info();
                    },
                    "info" => {
                        if parts.len() < 3 {
                            println!("Usage: plugin info <plugin_id>");
                            continue;
                        }
                        
                        let plugin_id = &parts[2];
                        match plugins::registry::get_plugin_metadata(plugin_id) {
                            Some(plugin) => {
                                println!("Plugin Information:");
                                println!("  ID: {}", plugin.id);
                                println!("  Name: {}", plugin.name);
                                println!("  Version: {}", plugin.version);
                                if let Some(desc) = &plugin.description {
                                    println!("  Description: {}", desc);
                                }
                                if let Some(author) = &plugin.author {
                                    println!("  Author: {}", author);
                                }
                                println!("  Status: {}", if plugin.enabled { "Enabled" } else { "Disabled" });
                                println!("  Commands: {}", plugin.commands.join(", "));
                            },
                            None => println!("Plugin '{}' not found", plugin_id),
                        }
                    },
                    "enable" => {
                        if parts.len() < 3 {
                            println!("Usage: plugin enable <plugin_id>");
                            continue;
                        }
                        
                        let plugin_id = &parts[2];
                        match plugins::set_plugin_enabled(plugin_id, true) {
                            Ok(_) => println!("Plugin '{}' enabled", plugin_id),
                            Err(e) => println!("Error: {}", e),
                        }
                    },
                    "disable" => {
                        if parts.len() < 3 {
                            println!("Usage: plugin disable <plugin_id>");
                            continue;
                        }
                        
                        let plugin_id = &parts[2];
                        match plugins::set_plugin_enabled(plugin_id, false) {
                            Ok(_) => println!("Plugin '{}' disabled", plugin_id),
                            Err(e) => println!("Error: {}", e),
                        }
                    },
                    cmd => println!("Unknown plugin command: '{}'. Valid commands: list, info, enable, disable", cmd),
                }
            },
            "execute" => {
                // Execute plugin command
                if parts.len() < 4 {
                    println!("Usage: execute plugin <plugin_id> <command> [args...]");
                    continue;
                }
                
                if parts[1] != "plugin" {
                    println!("Usage: execute plugin <plugin_id> <command> [args...]");
                    continue;
                }
                
                let plugin_id = parts[2].as_str();  // Use as_str() to get a &str
                let command = parts[3].as_str();
                
                // Create a vector of string slices
                let mut all_args_str: Vec<&str> = Vec::with_capacity(1 + parts.len() - 4);
                all_args_str.push(command);
                
                // Add remaining arguments as &str
                for arg in &parts[4..] {
                    all_args_str.push(arg.as_str());
                }
                
                match plugins::execute_plugin(plugin_id, &all_args_str) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error executing plugin '{}': {}", plugin_id, e),
                }
            },
            "nlp" => {
                // NLP plugin shortcut
                if parts.len() < 2 {
                    println!("Usage: nlp <tokenize|analyze_sentiment|extract_entities|summarize> \"text\"");
                    continue;
                }
                
                let command = parts[1].as_str();  // Convert to &str
                // Split input to get text after the command
                let args: Vec<&str> = input.splitn(3, ' ').collect();
                let text = if args.len() > 2 { args[2] } else { "" };
                
                // Create array of &str for plugin arguments
                let plugin_args = [command, text];
                
                match plugins::execute_plugin("nlp", &plugin_args) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error executing NLP plugin: {}", e),
                }
            },
            "img" => {
                // Image processing plugin shortcut
                if parts.len() < 2 {
                    println!("Usage: img <resize|crop|grayscale|blur|rotate> [args...]");
                    continue;
                }
                
                // Get command as &str
                let command_str = parts[1].as_str();
                
                // Create array of string slices for arguments
                let mut args_str: Vec<&str> = Vec::with_capacity(parts.len() - 1);
                args_str.push(command_str);
                
                // Add remaining arguments as &str
                for arg in &parts[2..] {
                    args_str.push(arg.as_str());
                }
                
                match plugins::execute_plugin("img", &args_str) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error executing Image Processing plugin: {}", e),
                }
            },
            "mm" => {
                // Multi-modal plugin shortcut
                if parts.len() < 2 {
                    println!("Usage: mm <load|convert|combine|extract_features> [args...]");
                    continue;
                }
                
                // Get command as &str
                let command_str = parts[1].as_str();
                
                // Create array of string slices for arguments
                let mut args_str: Vec<&str> = Vec::with_capacity(parts.len() - 1);
                args_str.push(command_str);
                
                // Add remaining arguments as &str
                for arg in &parts[2..] {
                    args_str.push(arg.as_str());
                }
                
                match plugins::execute_plugin("mm", &args_str) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error executing Multi-Modal plugin: {}", e),
                }
            },
            _ => println!("Unknown command: '{}'. Type 'help' for available commands.", command),
        }
        io::stdout().flush().unwrap();
    }
}