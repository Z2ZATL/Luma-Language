use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, Read, Cursor};
use std::sync::RwLock;
use std::collections::HashMap;
use lazy_static::lazy_static;

use image::{DynamicImage, ImageFormat, io::Reader as ImageReader};
use hound::WavReader;
use csv::ReaderBuilder;
use crate::ai::data::loaders;
use crate::ai::data::preprocessors;

/// Enum to represent different types of multi-modal data
#[derive(Debug, Clone)]
pub enum MultiModalData {
    /// CSV or tabular data (numerical values)
    Text(Vec<Vec<f64>>),
    
    /// Image data (stores dimensions and raw pixel data)
    Image {
        width: u32,
        height: u32,
        channels: u8,
        data: Vec<u8>,
    },
    
    /// Audio data (stores sample rate and raw audio samples)
    Audio {
        sample_rate: u32,
        channels: u16,
        samples: Vec<i16>,
    },
    
    /// JSON data (stores parsed values as strings)
    Json(HashMap<String, String>),
    
    /// Raw binary data
    Binary(Vec<u8>),
}

/// Metadata for multi-modal datasets
#[derive(Debug, Clone)]
pub struct MultiModalDataset {
    name: String,
    data_type: String,
    path: Option<String>,
    loaded: bool,
    data: Option<MultiModalData>,
    preprocessed: bool,
}

impl MultiModalDataset {
    pub fn get_name(&self) -> &str {
        &self.name
    }
    
    pub fn get_type(&self) -> &str {
        &self.data_type
    }
    
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
    
    pub fn get_data(&self) -> Option<&MultiModalData> {
        self.data.as_ref()
    }
    
    pub fn is_preprocessed(&self) -> bool {
        self.preprocessed
    }
}

lazy_static! {
    static ref MULTI_MODAL_DATASETS: RwLock<HashMap<String, MultiModalDataset>> = RwLock::new(HashMap::new());
}

/// Detect the file type from its path
fn detect_file_type(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("csv") | Some("tsv") | Some("txt") => Some("text"),
        Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") | Some("gif") => Some("image"),
        Some("wav") | Some("mp3") | Some("ogg") => Some("audio"),
        Some("json") => Some("json"),
        _ => None,
    }
}

/// Function to load multi-modal data based on file extension
pub fn load_multi_modal(path: &str) -> Result<MultiModalData, String> {
    let path = Path::new(path);
    
    match path.extension().and_then(|s| s.to_str()) {
        Some("csv") | Some("tsv") | Some("txt") => {
            let data = loaders::load_csv(path)?; // Load CSV data
            Ok(MultiModalData::Text(data))
        }
        Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") | Some("gif") => {
            let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
            let width = img.width();
            let height = img.height();
            
            // Convert to RGB format for consistency
            let rgb_img = img.to_rgb8();
            let channels = 3; // RGB has 3 channels
            
            // Flatten the pixel data
            let data = rgb_img.into_raw();
            
            Ok(MultiModalData::Image {
                width,
                height,
                channels,
                data,
            })
        }
        Some("wav") => {
            let reader = WavReader::open(path).map_err(|e| format!("Failed to open WAV file: {}", e))?;
            let spec = reader.spec();
            let sample_rate = spec.sample_rate;
            let channels = spec.channels;
            
            let samples: Vec<i16> = reader.into_samples()
                .filter_map(Result::ok)
                .collect();
                
            Ok(MultiModalData::Audio {
                sample_rate,
                channels,
                samples,
            })
        }
        Some("json") => {
            let file = File::open(path).map_err(|e| format!("Failed to open JSON file: {}", e))?;
            let reader = BufReader::new(file);
            let json: serde_json::Value = serde_json::from_reader(reader)
                .map_err(|e| format!("Failed to parse JSON: {}", e))?;
            
            // Extract string values for simple representation
            let mut map = HashMap::new();
            if let serde_json::Value::Object(obj) = json {
                for (key, value) in obj {
                    map.insert(key, value.to_string());
                }
            }
            
            Ok(MultiModalData::Json(map))
        }
        _ => {
            // Try to read as binary
            let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).map_err(|e| format!("Failed to read file: {}", e))?;
            
            // Return as binary data
            Ok(MultiModalData::Binary(buffer))
        }
    }
}

/// Load multi-modal data and store with a name
pub fn load_dataset(path: &str, name: &str) -> Result<(), String> {
    // Check if dataset name already exists
    let datasets = MULTI_MODAL_DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
    if datasets.contains_key(name) {
        return Err(format!("Multi-modal dataset name '{}' already exists", name));
    }
    drop(datasets); // Release read lock
    
    // Detect file type
    let path_obj = Path::new(path);
    let data_type = detect_file_type(path_obj)
        .ok_or_else(|| format!("Unsupported file type for path: {}", path))?
        .to_string();
    
    // Load the data
    let data = load_multi_modal(path)?;
    
    // Create the dataset entry
    let dataset = MultiModalDataset {
        name: name.to_string(),
        data_type,
        path: Some(path.to_string()),
        loaded: true,
        data: Some(data),
        preprocessed: false,
    };
    
    // Store the dataset
    let mut datasets = MULTI_MODAL_DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.insert(name.to_string(), dataset);
    
    println!("Loaded multi-modal dataset '{}' from '{}'", name, path);
    Ok(())
}

/// Get a loaded multi-modal dataset
pub fn get_dataset(name: &str) -> Option<MultiModalDataset> {
    let datasets = MULTI_MODAL_DATASETS.read().ok()?;
    datasets.get(name).cloned()
}

/// Convert image data to a numerical matrix (for ML use)
pub fn image_to_matrix(image_data: &MultiModalData) -> Result<Vec<Vec<f64>>, String> {
    match image_data {
        MultiModalData::Image { width, height, channels, data } => {
            let mut matrix = Vec::with_capacity(*height as usize);
            
            // Process each row
            for y in 0..*height {
                let mut row = Vec::with_capacity(*width as usize * *channels as usize);
                
                // Process each pixel in the row
                for x in 0..*width {
                    let pos = ((y * *width + x) * (*channels as u32)) as usize;
                    
                    // Add each channel value
                    for c in 0..*channels {
                        if pos + (c as usize) < data.len() {
                            row.push(data[pos + (c as usize)] as f64 / 255.0); // Normalize to [0,1]
                        }
                    }
                }
                
                matrix.push(row);
            }
            
            Ok(matrix)
        },
        _ => Err("Data is not an image".to_string()),
    }
}

/// Convert audio data to a numerical matrix (for ML use)
pub fn audio_to_matrix(audio_data: &MultiModalData) -> Result<Vec<Vec<f64>>, String> {
    match audio_data {
        MultiModalData::Audio { sample_rate: _, channels: _, samples } => {
            // Determine frame size (number of samples per frame)
            let frame_size = 1024; // Typical frame size for audio processing
            let num_frames = samples.len() / frame_size;
            
            let mut matrix = Vec::with_capacity(num_frames);
            
            // Process each frame
            for i in 0..num_frames {
                let start_idx = i * frame_size;
                let end_idx = (start_idx + frame_size).min(samples.len());
                
                // Extract and normalize the frame data
                let frame: Vec<f64> = samples[start_idx..end_idx]
                    .iter()
                    .map(|&s| s as f64 / 32768.0) // Normalize to [-1, 1]
                    .collect();
                
                matrix.push(frame);
            }
            
            Ok(matrix)
        },
        _ => Err("Data is not audio".to_string()),
    }
}

/// Apply preprocessing to multi-modal data
pub fn preprocess_dataset(name: &str, output_name: Option<&str>) -> Result<String, String> {
    // Get the source dataset
    let source = match get_dataset(name) {
        Some(ds) => ds,
        None => return Err(format!("Multi-modal dataset '{}' not found", name)),
    };
    
    if !source.loaded {
        return Err(format!("Dataset '{}' is not loaded", name));
    }
    
    let source_data = match source.data {
        Some(ref data) => data.clone(),
        None => return Err(format!("Dataset '{}' has no data", name)),
    };
    
    // Determine output name
    let output = output_name.map(|s| s.to_string()).unwrap_or_else(|| format!("{}_preprocessed", name));
    
    // Check if output name already exists
    let datasets = MULTI_MODAL_DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
    if datasets.contains_key(&output) {
        return Err(format!("Dataset name '{}' already exists", output));
    }
    drop(datasets);
    
    // Preprocess based on data type
    let (processed_data, matrix_data) = match source_data {
        MultiModalData::Text(data) => {
            // For text (numerical) data, apply normalization
            let normalized = preprocessors::normalize_data_clone(&data);
            (MultiModalData::Text(normalized.clone()), Some(normalized))
        },
        MultiModalData::Image { .. } => {
            // Convert image to matrix, normalize, then keep both forms
            let matrix = image_to_matrix(&source_data)?;
            let normalized = preprocessors::normalize_data_clone(&matrix);
            (source_data, Some(normalized))
        },
        MultiModalData::Audio { .. } => {
            // Convert audio to matrix, normalize, then keep both forms
            let matrix = audio_to_matrix(&source_data)?;
            let normalized = preprocessors::normalize_data_clone(&matrix);
            (source_data, Some(normalized))
        },
        _ => (source_data, None),
    };
    
    // If we have matrix data, also store it as a regular dataset
    if let Some(matrix) = matrix_data {
        // Create a standard dataset with the normalized data
        let mut labels = Vec::new();
        for _ in 0..matrix.len() {
            labels.push(vec![0.0]); // Placeholder labels
        }
        
        // This lets us use standard ML tools with the data
        let dataset_meta = loaders::DatasetMetadata {
            name: output.clone(),
            path: None,
            headers: None,
            data: matrix,
            labels,
            lazy: false,
            loaded: true,
            label_column: None,
        };
        
        // Store in the regular dataset collection
        let datasets_result = loaders::DATASETS.write();
        if let Ok(mut datasets) = datasets_result {
            datasets.insert(output.clone(), dataset_meta);
        }
    }
    
    // Store the processed dataset
    let mut datasets = MULTI_MODAL_DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.insert(output.clone(), MultiModalDataset {
        name: output.clone(),
        data_type: source.data_type,
        path: None,
        loaded: true,
        data: Some(processed_data),
        preprocessed: true,
    });
    
    println!("Preprocessed multi-modal dataset '{}' and saved as '{}'", name, output);
    Ok(output)
}

/// List all available multi-modal datasets
pub fn list_datasets() {
    let datasets = match MULTI_MODAL_DATASETS.read() {
        Ok(d) => d,
        Err(e) => {
            println!("Error listing multi-modal datasets: {}", e);
            return;
        }
    };
    
    if datasets.is_empty() {
        println!("No multi-modal datasets available.");
        return;
    }
    
    println!("Available multi-modal datasets:");
    println!("{:<20} {:<15} {:<10} {:<15}", "Name", "Type", "Loaded", "Preprocessed");
    println!("{}", "-".repeat(60));
    
    for (name, ds) in datasets.iter() {
        println!("{:<20} {:<15} {:<10} {:<15}", 
            name, 
            ds.data_type, 
            if ds.loaded { "Yes" } else { "No" },
            if ds.preprocessed { "Yes" } else { "No" }
        );
    }
}

/// Clear all multi-modal datasets
pub fn clear_datasets() {
    if let Ok(mut datasets) = MULTI_MODAL_DATASETS.write() {
        datasets.clear();
        println!("Cleared all multi-modal datasets.");
    } else {
        println!("Failed to acquire lock for multi-modal dataset cleanup.");
    }
}