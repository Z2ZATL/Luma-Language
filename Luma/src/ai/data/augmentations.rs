use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::ReaderBuilder;
use lazy_static::lazy_static;
use std::collections::HashMap;
use prettytable::{Table, Row, Cell};
use crate::ai::data::preprocessors::{normalize_data_clone, scale_data_clone};

// Data augmentation methods
#[derive(Debug, Clone, Copy)]
pub enum AugmentationMethod {
    // Preprocessing-based augmentations
    Normalize,
    Scale(f64, f64),    // min, max
    
    // Data augmentation techniques
    Noise(f64),         // Add Gaussian noise with std deviation
    Dropout(f64),       // Randomly drop values with probability
    Rotate90,           // Rotate 2D data by 90 degrees
    Mirror,             // Mirror data along an axis
    Shuffle,            // Shuffle rows
}

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
    path: Option<String>,         // Store the path for lazy loading
    headers: Option<Vec<String>>, // Store headers if present
    data: Option<Vec<Vec<f64>>>,  // Lazy-loaded data
    source_dataset: Option<String>, // Name of the source dataset (if this is an augmented version)
    method: Option<String>,       // Method used for augmentation
    lazy: bool,                   // Whether this dataset is lazy loaded
    loaded: bool,                 // Whether data has been loaded
}

impl DatasetMetadata {
    pub fn get_data(&self) -> Option<&Vec<Vec<f64>>> {
        self.data.as_ref()
    }
    
    pub fn get_name(&self) -> &str {
        &self.name
    }
    
    pub fn get_headers(&self) -> Option<&Vec<String>> {
        self.headers.as_ref()
    }
    
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
    
    pub fn is_lazy(&self) -> bool {
        self.lazy
    }
    
    pub fn get_method(&self) -> Option<&str> {
        self.method.as_ref().map(|s| s.as_str())
    }
}

lazy_static! {
    static ref AUGMENTED_DATASETS: RwLock<HashMap<String, DatasetMetadata>> = RwLock::new(HashMap::new());
}

/// Augment a dataset from a file
/// 
/// # Arguments
/// * `path` - Path to the CSV file
/// * `name` - Name to assign to the augmented dataset
/// * `lazy` - Whether to load data lazily (default: false)
/// * `method` - Augmentation method to apply
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` with error message on failure
pub fn augment_dataset_from_file(
    path: &str, 
    name: &str, 
    lazy: bool, 
    method: Option<AugmentationMethod>
) -> Result<(), String> {
    let mut datasets = AUGMENTED_DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    
    // Check for duplicate name
    if datasets.contains_key(name) {
        return Err(format!("Augmentation name '{}' already exists", name));
    }

    let (headers, data, loaded) = if !lazy {
        // Immediate loading
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let buf_reader = BufReader::new(file);
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(buf_reader);
            
        let mut data = Vec::new();
        let mut headers = None;
        let mut row_count = 0;

        // Read headers
        match reader.headers() {
            Ok(header_record) => {
                headers = Some(header_record.iter().map(|h| h.to_string()).collect());
            }
            Err(e) => println!("Warning: Failed to read headers: {}. Proceeding without headers.", e),
        }

        // Read data rows
        for result in reader.records() {
            row_count += 1;
            let record = result.map_err(|e| format!("Failed to read CSV at row {}: {}", row_count, e))?;
            let row: Vec<f64> = record.iter()
                .filter_map(|field| field.parse::<f64>().ok())
                .collect();
            if !row.is_empty() {
                data.push(row);
            } else {
                println!("Debug: Empty row detected at row {}", row_count);
            }
        }
        
        if data.is_empty() {
            return Err(format!("No valid data found in the CSV file. Processed {} rows.", row_count));
        }
        
        println!("Debug: Loaded {} rows into data (augment_dataset)", data.len());
        
        // Apply augmentation method if specified
        let augmented_data = if let Some(method) = method {
            apply_augmentation(&data, method)?
        } else {
            data
        };
        
        (headers, Some(augmented_data), true)
    } else {
        // Lazy loading - will load data when accessed
        (None, None, false)
    };

    // Store the method used for augmentation
    let method_str = method.map(|m| format!("{:?}", m));

    // Create and store the dataset
    datasets.insert(
        name.to_string(),
        DatasetMetadata {
            name: name.to_string(),
            path: Some(path.to_string()),
            headers,
            data,
            source_dataset: None,
            method: method_str,
            lazy,
            loaded,
        },
    );
    
    Ok(())
}

/// Augment from an existing dataset
pub fn augment_dataset(
    source_name: &str, 
    new_name: &str, 
    method: AugmentationMethod
) -> Result<(), String> {
    // First check if source exists in our augmented datasets
    let source_in_augmented = {
        let datasets = AUGMENTED_DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
        datasets.contains_key(source_name)
    };
    
    // Get the source data
    let (source_data, source_headers) = if source_in_augmented {
        // Source is in our augmented datasets
        let source = get_augmentation(source_name)
            .ok_or_else(|| format!("Source dataset '{}' not found", source_name))?;
            
        let data = source.data
            .as_ref()
            .ok_or_else(|| format!("Source dataset '{}' has no data", source_name))?;
            
        (data.clone(), source.headers.clone())
    } else {
        // Try to get from regular datasets
        use crate::ai::data::loaders;
        let source = loaders::get_dataset(source_name)
            .ok_or_else(|| format!("Source dataset '{}' not found", source_name))?;
            
        (source.get_data().clone(), source.get_headers().clone())
    };
    
    // Check for duplicate name
    {
        let datasets = AUGMENTED_DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
        if datasets.contains_key(new_name) {
            return Err(format!("Augmentation name '{}' already exists", new_name));
        }
    }
    
    // Apply the augmentation
    let augmented_data = apply_augmentation(&source_data, method)?;
    
    // Store the new augmented dataset
    let mut datasets = AUGMENTED_DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.insert(
        new_name.to_string(),
        DatasetMetadata {
            name: new_name.to_string(),
            path: None,
            headers: source_headers,
            data: Some(augmented_data),
            source_dataset: Some(source_name.to_string()),
            method: Some(format!("{:?}", method)),
            lazy: false,
            loaded: true,
        },
    );
    
    println!("Created augmented dataset '{}' from '{}' using method {:?}", 
        new_name, source_name, method);
    
    Ok(())
}

/// Apply augmentation method to data
fn apply_augmentation(data: &[Vec<f64>], method: AugmentationMethod) -> Result<Vec<Vec<f64>>, String> {
    if data.is_empty() {
        return Err("Cannot augment empty dataset".to_string());
    }
    
    match method {
        AugmentationMethod::Normalize => {
            // Use normalize_data_clone from preprocessors.rs
            Ok(normalize_data_clone(data))
        },
        AugmentationMethod::Scale(min, max) => {
            // Use scale_data_clone from preprocessors.rs
            Ok(scale_data_clone(data, min, max))
        },
        AugmentationMethod::Noise(std_dev) => {
            // Add Gaussian noise
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut result = data.to_vec();
            
            for row in result.iter_mut() {
                for val in row.iter_mut() {
                    // Add Gaussian noise
                    *val += rng.sample::<f64, _>(rand_distr::StandardNormal) * std_dev;
                }
            }
            
            Ok(result)
        },
        AugmentationMethod::Dropout(prob) => {
            // Randomly set values to 0 with probability prob
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut result = data.to_vec();
            
            for row in result.iter_mut() {
                for val in row.iter_mut() {
                    if rng.gen::<f64>() < prob {
                        *val = 0.0;
                    }
                }
            }
            
            Ok(result)
        },
        AugmentationMethod::Rotate90 => {
            // Only works for 2D data with equal dimensions
            let row_count = data.len();
            if row_count == 0 {
                return Ok(Vec::new());
            }
            
            let col_count = data[0].len();
            if row_count != col_count {
                return Err(format!(
                    "Rotate90 requires square data matrix, got {}x{}", 
                    row_count, col_count
                ));
            }
            
            let mut result = vec![vec![0.0; row_count]; col_count];
            
            // Rotate 90 degrees clockwise
            for i in 0..row_count {
                for j in 0..col_count {
                    result[j][row_count - 1 - i] = data[i][j];
                }
            }
            
            Ok(result)
        },
        AugmentationMethod::Mirror => {
            // Mirror across vertical axis
            let mut result = data.to_vec();
            
            for row in result.iter_mut() {
                row.reverse();
            }
            
            Ok(result)
        },
        AugmentationMethod::Shuffle => {
            // Shuffle the rows
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut result = data.to_vec();
            result.shuffle(&mut rng);
            
            Ok(result)
        },
    }
}

/// Get an augmented dataset by name, loading it first if it's lazy
pub fn get_augmentation(name: &str) -> Option<DatasetMetadata> {
    // Try to get the dataset
    let datasets_read = AUGMENTED_DATASETS.read().map_err(|e| {
        println!("Lock error when reading datasets: {}", e);
        e
    }).ok()?;
    
    // If dataset exists and is already loaded (or not lazy), return a clone
    if let Some(dataset) = datasets_read.get(name) {
        if dataset.loaded || !dataset.lazy {
            return Some(dataset.clone());
        }
    }
    
    // If dataset exists but needs lazy loading, drop read lock and acquire write lock
    let exists = datasets_read.contains_key(name);
    drop(datasets_read);
    
    if exists {
        let mut datasets_write = AUGMENTED_DATASETS.write().map_err(|e| {
            println!("Lock error when writing datasets: {}", e);
            e
        }).ok()?;
        
        // Get mutable reference to the dataset
        if let Some(dataset) = datasets_write.get_mut(name) {
            // Check again if it needs loading (might have been loaded while we were waiting for the lock)
            if !dataset.loaded && dataset.lazy {
                // Load the data from path
                if let Some(path) = &dataset.path {
                    let file = File::open(path).map_err(|e| {
                        println!("Failed to open file {}: {}", path, e);
                        e
                    }).ok()?;
                    
                    let buf_reader = BufReader::new(file);
                    let mut reader = ReaderBuilder::new()
                        .has_headers(true)
                        .trim(csv::Trim::All)
                        .from_reader(buf_reader);
                        
                    let mut data = Vec::new();
                    let mut headers = None;
                    let mut row_count = 0;
                    
                    // Read headers
                    match reader.headers() {
                        Ok(header_record) => {
                            headers = Some(header_record.iter().map(|h| h.to_string()).collect());
                        }
                        Err(e) => println!("Warning: Failed to read headers: {}. Proceeding without headers.", e),
                    }
                    
                    // Read data rows
                    for result in reader.records() {
                        row_count += 1;
                        if let Ok(record) = result {
                            let row: Vec<f64> = record.iter()
                                .filter_map(|field| field.parse::<f64>().ok())
                                .collect();
                            if !row.is_empty() {
                                data.push(row);
                            } else {
                                println!("Debug: Empty row detected at row {} (get_augmentation)", row_count);
                            }
                        }
                    }
                    
                    if !data.is_empty() {
                        println!("Debug: Loaded {} rows into data (get_augmentation)", data.len());
                        
                        // Apply augmentation method if specified
                        if let Some(method_str) = &dataset.method {
                            // This is a simplified version - ideally we would parse the method string
                            let normalized = normalize_data_clone(&data);
                            dataset.data = Some(normalized);
                        } else {
                            dataset.data = Some(data);
                        }
                        
                        dataset.headers = headers;
                        dataset.loaded = true;
                    }
                }
            }
            
            // Return a clone
            return Some(dataset.clone());
        }
    }
    
    None
}

/// Print an augmented dataset in tabular format
pub fn print_augmentation(name: &str) {
    if let Some(dataset) = get_augmentation(name) {
        println!("Augmentation Name: {}", dataset.name);
        
        if let Some(source) = &dataset.source_dataset {
            println!("Source Dataset: {}", source);
        }
        
        if let Some(method) = &dataset.method {
            println!("Augmentation Method: {}", method);
        }
        
        if !dataset.loaded {
            println!("Dataset is configured for lazy loading but hasn't been loaded yet.");
            return;
        }
        
        if let Some(data) = &dataset.data {
            // Create a table to display the data
            let mut table = Table::new();
            // Add header
            let headers: Vec<Cell> = match &dataset.headers {
                Some(h) => h.iter().map(|h| Cell::new(h)).collect(),
                None => (0..data[0].len())
                    .map(|i| Cell::new(&format!("Col {}", i)))
                    .collect(),
            };
            table.add_row(Row::new(headers));
            
            // Add data rows (limit to 10 rows if there are too many)
            let max_rows = 10.min(data.len());
            for i in 0..max_rows {
                let cells: Vec<Cell> = data[i].iter()
                    .map(|val| Cell::new(&format!("{:.3}", val)))
                    .collect();
                table.add_row(Row::new(cells));
            }
            
            // Show ellipsis if there are more rows
            if data.len() > max_rows {
                let ellipsis_row = Row::new(vec![Cell::new("..."); headers.len()]);
                table.add_row(ellipsis_row);
                println!("(Showing first {} of {} rows)", max_rows, data.len());
            }
            
            table.printstd();
        } else {
            println!("Data: None");
        }
    } else {
        println!("Augmentation {} not found", name);
    }
}

/// List all available augmented datasets
pub fn list_augmentations() {
    let datasets = match AUGMENTED_DATASETS.read() {
        Ok(d) => d,
        Err(e) => {
            println!("Error listing augmentations: {}", e);
            return;
        }
    };
    
    if datasets.is_empty() {
        println!("No augmented datasets available.");
        return;
    }
    
    println!("Available augmented datasets:");
    let mut table = Table::new();
    table.add_row(Row::new(vec![
        Cell::new("Name"),
        Cell::new("Source"),
        Cell::new("Method"),
        Cell::new("Rows"),
        Cell::new("Loaded"),
        Cell::new("Lazy")
    ]));
    
    for (name, ds) in datasets.iter() {
        let rows = if ds.loaded {
            if let Some(data) = &ds.data {
                data.len().to_string()
            } else {
                "0".to_string()
            }
        } else {
            "?".to_string()
        };
        
        let source = ds.source_dataset.as_ref().map_or("-".to_string(), |s| s.clone());
        let method = ds.method.as_ref().map_or("-".to_string(), |m| m.clone());
        
        table.add_row(Row::new(vec![
            Cell::new(name),
            Cell::new(&source),
            Cell::new(&method),
            Cell::new(&rows),
            Cell::new(if ds.loaded { "Yes" } else { "No" }),
            Cell::new(if ds.lazy { "Yes" } else { "No" })
        ]));
    }
    
    table.printstd();
}

/// Clear all augmentations
pub fn clear_augmentations() {
    if let Ok(mut datasets) = AUGMENTED_DATASETS.write() {
        datasets.clear();
        println!("Cleared all augmented datasets.");
    } else {
        println!("Failed to acquire lock for augmentation cleanup.");
    }
}