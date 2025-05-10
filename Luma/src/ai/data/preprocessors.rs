use crate::ai::data::loaders::DatasetMetadata;

/// Preprocessing methods for data
#[derive(Debug, Clone, Copy)]
pub enum PreprocessingMethod {
    /// Normalize data (mean=0, std=1)
    Normalize,
    /// Min-max scaling to [0,1]
    MinMaxScale,
    /// Min-max scaling to custom range
    MinMaxScaleRange(f64, f64),
    /// Log transformation (natural log)
    LogTransform,
    /// Square root transformation
    SqrtTransform,
}

/// Normalizes a dataset by subtracting the mean and dividing by the standard deviation
/// 
/// This function works on a mutable slice of data vectors.
pub fn normalize_data(data: &mut [Vec<f64>]) {
    for vec_row in data {
        let sum: f64 = vec_row.iter().sum();
        let mean = sum / vec_row.len() as f64;
        let variance: f64 = vec_row.iter().map(|x| (x - mean).powi(2)).sum();
        let std_dev = (variance / vec_row.len() as f64).sqrt();

        for val in vec_row.iter_mut() {
            if std_dev != 0.0 {
                *val = (*val - mean) / std_dev;
            } else {
                *val = *val - mean; // Avoid division by zero
            }
        }
    }
}

/// Normalizes data by returning a new normalized vector without modifying the input
/// 
/// This version is used by augmentations.rs where input data shouldn't be modified.
pub fn normalize_data_clone(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = data.to_vec();
    normalize_data(&mut result);
    result
}

/// Preprocesses a dataset by applying normalization
pub fn preprocess(data: &mut Vec<Vec<f64>>) {
    let data_slice = &mut data[..];
    normalize_data(data_slice);
}

/// Scales data to a specific range (min-max scaling)
pub fn scale_data(data: &mut [Vec<f64>], min: f64, max: f64) {
    for vec_row in data {
        let row_min = *vec_row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let row_max = *vec_row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&1.0);
        let range = row_max - row_min;

        if range != 0.0 {
            for val in vec_row.iter_mut() {
                *val = min + (*val - row_min) * (max - min) / range;
            }
        }
    }
}

/// Scales data to a range without modifying the input
pub fn scale_data_clone(data: &[Vec<f64>], min: f64, max: f64) -> Vec<Vec<f64>> {
    let mut result = data.to_vec();
    scale_data(&mut result, min, max);
    result
}

/// Applies log transformation (natural log) to data
/// Adds a small constant to handle zeros
pub fn log_transform(data: &mut [Vec<f64>]) {
    const EPSILON: f64 = 1e-10;
    
    for vec_row in data {
        for val in vec_row.iter_mut() {
            // Add small constant to handle zeros
            *val = (*val + EPSILON).ln();
        }
    }
}

/// Applies square root transformation to data
pub fn sqrt_transform(data: &mut [Vec<f64>]) {
    for vec_row in data {
        for val in vec_row.iter_mut() {
            if *val >= 0.0 {
                *val = val.sqrt();
            } else {
                *val = 0.0; // Handle negative values by setting to 0
            }
        }
    }
}

/// Preprocess a dataset using the specified method
pub fn preprocess_dataset(
    dataset_name: &str, 
    method: PreprocessingMethod, 
    new_name: Option<&str>
) -> Result<String, String> {
    use crate::ai::data::loaders;
    
    // Get the original dataset
    let dataset = match loaders::get_dataset(dataset_name) {
        Some(ds) => ds,
        None => return Err(format!("Dataset '{}' not found", dataset_name)),
    };
    
    if !dataset.is_loaded() {
        return Err(format!("Dataset '{}' is not loaded", dataset_name));
    }
    
    // Clone the data
    let mut processed_data = dataset.get_data().clone();
    
    // Apply the preprocessing method
    match method {
        PreprocessingMethod::Normalize => {
            normalize_data(&mut processed_data);
        },
        PreprocessingMethod::MinMaxScale => {
            scale_data(&mut processed_data, 0.0, 1.0);
        },
        PreprocessingMethod::MinMaxScaleRange(min, max) => {
            scale_data(&mut processed_data, min, max);
        },
        PreprocessingMethod::LogTransform => {
            log_transform(&mut processed_data);
        },
        PreprocessingMethod::SqrtTransform => {
            sqrt_transform(&mut processed_data);
        },
    }
    
    // Determine the new dataset name
    let output_name = match new_name {
        Some(name) => name.to_string(),
        None => {
            let method_suffix = match method {
                PreprocessingMethod::Normalize => "norm",
                PreprocessingMethod::MinMaxScale => "scaled",
                PreprocessingMethod::MinMaxScaleRange(_, _) => "scaled_custom",
                PreprocessingMethod::LogTransform => "log",
                PreprocessingMethod::SqrtTransform => "sqrt",
            };
            format!("{}_{}", dataset_name, method_suffix)
        }
    };
    
    // Create a new dataset with the preprocessed data
    let dataset_labels = dataset.get_labels().clone();
    let dataset_headers = dataset.get_headers().clone();
    
    // Store the new dataset
    let new_dataset = DatasetMetadata {
        name: output_name.clone(),
        path: None,
        headers: dataset_headers,
        data: processed_data,
        labels: dataset_labels,
        lazy: false,
        loaded: true,
        label_column: None,
    };
    
    // Insert the new dataset into storage
    match crate::ai::data::loaders::DATASETS.write() {
        Ok(mut datasets) => {
            if datasets.contains_key(&output_name) {
                return Err(format!("Dataset name '{}' already exists", output_name));
            }
            datasets.insert(output_name.clone(), new_dataset);
        },
        Err(e) => return Err(format!("Failed to acquire lock: {}", e)),
    }
    
    println!("Preprocessed dataset '{}' with method '{:?}' and saved as '{}'", 
             dataset_name, method, output_name);
    
    Ok(output_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_data() {
        let mut data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        normalize_data(&mut data);
        for row in data {
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!((mean - 0.0).abs() < 1e-10, "Mean should be close to 0");
        }
    }

    #[test]
    fn test_scale_data() {
        let mut data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        scale_data(&mut data, 0.0, 1.0);
        for row in data {
            let min = row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!((min - 0.0).abs() < 1e-10, "Min should be 0");
            assert!((max - 1.0).abs() < 1e-10, "Max should be 1");
        }
    }
    
    #[test]
    fn test_clone_methods() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        
        // Test normalize clone
        let normalized = normalize_data_clone(&data);
        assert_eq!(data.len(), normalized.len());
        assert_eq!(data[0].len(), normalized[0].len());
        
        // Original data should be unchanged
        assert_eq!(data[0][0], 1.0);
        
        // Test scale clone
        let scaled = scale_data_clone(&data, -1.0, 1.0);
        assert_eq!(data.len(), scaled.len());
        
        // Original data should still be unchanged
        assert_eq!(data[0][0], 1.0);
    }
    
    #[test]
    fn test_log_transform() {
        let mut data = vec![vec![1.0, 2.0, 3.0]];
        let original = data.clone();
        
        log_transform(&mut data);
        
        // Check values roughly match ln(x)
        assert!((data[0][0] - original[0][0].ln()).abs() < 1e-9);
        assert!((data[0][1] - original[0][1].ln()).abs() < 1e-9);
    }
    
    #[test]
    fn test_sqrt_transform() {
        let mut data = vec![vec![1.0, 4.0, 9.0, -1.0]];
        
        sqrt_transform(&mut data);
        
        // Check values match sqrt(x)
        assert!((data[0][0] - 1.0).abs() < 1e-9);
        assert!((data[0][1] - 2.0).abs() < 1e-9);
        assert!((data[0][2] - 3.0).abs() < 1e-9);
        assert!((data[0][3] - 0.0).abs() < 1e-9); // Negative value should be 0
    }
}