use std::path::Path;
use image::DynamicImage;
use hound::WavReader;
use crate::ai::data::loaders; // Assuming loaders.rs has load_csv

// Enum to represent different types of multi-modal data
pub enum MultiModalData {
    Text(Vec<Vec<f64>>),  // For CSV or tabular data
    Image(DynamicImage),  // For image data (PNG, JPG)
    Audio(Vec<i16>),      // For audio data (WAV)
}

// Function to load multi-modal data based on file extension
pub fn load_multi_modal(path: &str) -> Result<MultiModalData, String> {
    let path = Path::new(path);
    match path.extension().and_then(|s| s.to_str()) {
        Some("csv") => {
            let data = loaders::load_csv(path)?; // Load CSV data
            Ok(MultiModalData::Text(data))
        }
        Some("png") | Some("jpg") => {
            let img = image::open(path).map_err(|e| e.to_string())?; // Load image
            Ok(MultiModalData::Image(img))
        }
        Some("wav") => {
            let reader = WavReader::open(path).map_err(|e| e.to_string())?; // Load WAV
            let samples: Vec<i16> = reader.into_samples().filter_map(Result::ok).collect();
            Ok(MultiModalData::Audio(samples))
        }
        _ => Err("Unsupported file type".to_string()), // Error for unrecognized types
    }
}