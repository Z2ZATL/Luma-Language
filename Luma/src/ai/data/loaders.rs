use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::ReaderBuilder;
use lazy_static::lazy_static;
use prettytable::{Table, Row, Cell};
use std::path::Path;
use std::collections::HashMap;
use crate::ai::data::preprocessors;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: String,
    pub path: Option<String>,          // Store file path for lazy loading
    pub headers: Option<Vec<String>>,
    pub data: Vec<Vec<f64>>,           // Features only
    pub labels: Vec<Vec<f64>>,         // Labels from the last column (as Vec<f64> for compatibility)
    pub lazy: bool,                    // Whether this dataset is lazily loaded
    pub loaded: bool,                  // Whether data has been loaded yet
    pub label_column: Option<usize>,   // Which column contains labels (default: last)
}

impl DatasetMetadata {
    pub fn get_data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    pub fn get_labels(&self) -> &Vec<Vec<f64>> {
        &self.labels
    }
    
    pub fn get_feature_count(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }
    
    pub fn get_name(&self) -> &str {
        &self.name
    }
    
    pub fn get_headers(&self) -> &Option<Vec<String>> {
        &self.headers
    }
    
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
}

lazy_static! {
    pub static ref DATASETS: RwLock<HashMap<String, DatasetMetadata>> = RwLock::new(HashMap::new());
}

/// Loads a dataset from a CSV file
///
/// # Arguments
/// * `path` - Path to the CSV file
/// * `name` - Name to assign to the dataset
/// * `lazy` - Whether to load data lazily (default: false)
/// * `label_col` - Which column contains labels (default: last column)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` with error message on failure
pub fn load_dataset(path: &str, name: &str, lazy: bool, label_col: Option<usize>) -> Result<(), String> {
    // Check if dataset name already exists
    let datasets = DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
    if datasets.contains_key(name) {
        return Err(format!("Dataset name '{}' already exists", name));
    }
    drop(datasets); // Release read lock
    
    // Create dataset metadata
    let mut metadata = DatasetMetadata {
        name: name.to_string(),
        path: Some(path.to_string()),
        headers: None,
        data: Vec::new(),
        labels: Vec::new(),
        lazy,
        loaded: false,
        label_column: label_col,
    };
    
    // If not lazy loading, load the data immediately
    if !lazy {
        load_dataset_data(&mut metadata)?;
    }
    
    // Save the metadata
    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.insert(name.to_string(), metadata);
    
    Ok(())
}

/// Internal function to load dataset data from file
fn load_dataset_data(metadata: &mut DatasetMetadata) -> Result<(), String> {
    let path = match &metadata.path {
        Some(p) => p,
        None => return Err("Dataset has no file path".to_string()),
    };
    
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let buf_reader = BufReader::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(buf_reader);

    let mut data = Vec::new();
    let mut labels = Vec::new();
    let mut row_count = 0;

    // Read headers
    match reader.headers() {
        Ok(header_record) => {
            metadata.headers = Some(header_record.iter().map(|h| h.to_string()).collect());
        }
        Err(e) => println!("Warning: Failed to read headers: {}. Proceeding without headers.", e),
    }
    
    // Determine label column index
    let label_index = match metadata.label_column {
        Some(col) => col,
        None => {
            // If no label column is specified, use the last column
            if let Some(headers) = &metadata.headers {
                headers.len() - 1
            } else {
                // If no headers, assume at least 2 columns and use the last one
                1
            }
        }
    };

    // Read records
    for result in reader.records() {
        row_count += 1;
        let record = result.map_err(|e| format!("Failed to read CSV at row {}: {}", row_count, e))?;
        let values: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        
        if values.len() <= label_index {
            println!("Debug: Skipping row {} due to insufficient columns", row_count);
            continue;
        }
        
        // Extract features (all columns except label column)
        let mut features: Vec<f64> = Vec::new();
        for (i, val) in values.iter().enumerate() {
            if i != label_index {
                if let Ok(num) = val.parse::<f64>() {
                    features.push(num);
                } else {
                    println!("Debug: Skipping non-numeric value '{}' in row {}, column {}", val, row_count, i);
                }
            }
        }
        
        // Extract label
        if let Ok(label) = values[label_index].parse::<f64>() {
            if !features.is_empty() {
                data.push(features);
                labels.push(vec![label]); // Wrap label in a Vec for compatibility with trainers.rs
            }
        } else {
            println!("Debug: Skipping row {} due to non-numeric label", row_count);
        }
    }

    // Check if any valid data was loaded
    if data.is_empty() {
        return Err(format!("No valid data found in the CSV file. Processed {} rows.", row_count));
    }

    println!("Debug: Loaded {} rows from {}", data.len(), path);
    
    // Update metadata
    metadata.data = data;
    metadata.labels = labels;
    metadata.loaded = true;
    
    Ok(())
}

/// Gets a dataset by name, loading it first if it's lazy and not loaded yet
pub fn get_dataset(name: &str) -> Option<DatasetMetadata> {
    // Try to get the dataset
    let datasets_read = DATASETS.read().map_err(|e| {
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
        let mut datasets_write = DATASETS.write().map_err(|e| {
            println!("Lock error when writing datasets: {}", e);
            e
        }).ok()?;
        
        // Get mutable reference to the dataset
        if let Some(dataset) = datasets_write.get_mut(name) {
            // Check again if it needs loading (might have been loaded while we were waiting for the lock)
            if !dataset.loaded && dataset.lazy {
                // Load the data
                if let Err(e) = load_dataset_data(dataset) {
                    println!("Error loading dataset '{}': {}", name, e);
                    return None;
                }
            }
            // Return a clone
            return Some(dataset.clone());
        }
    }
    
    None
}

/// Prints a dataset in tabular format
pub fn print_dataset(name: &str) {
    if let Some(dataset) = get_dataset(name) {
        println!("Dataset Name: {}", dataset.name);
        
        if !dataset.loaded {
            println!("Dataset is configured for lazy loading but hasn't been loaded yet.");
            return;
        }
        
        // Create a pretty table
        let mut table = Table::new();
        
        // Add headers
        let headers: Vec<Cell> = match &dataset.headers {
            Some(h) => h.iter().map(|h| Cell::new(h)).collect(),
            None => (0..dataset.data[0].len() + 1)
                .map(|i| Cell::new(&format!("Col {}", i)))
                .collect(),
        };
        table.add_row(Row::new(headers.clone()));
        
        // Add data rows (limit to 10 rows if there are too many)
        let max_rows = 10.min(dataset.data.len());
        for i in 0..max_rows {
            let mut cells: Vec<Cell> = dataset.data[i].iter()
                .map(|val| Cell::new(&format!("{:.2}", val)))
                .collect();
            cells.push(Cell::new(&format!("{:.2}", dataset.labels[i][0])));
            table.add_row(Row::new(cells));
        }
        
        // Show ellipsis if there are more rows
        if dataset.data.len() > max_rows {
            let ellipsis_row = Row::new(vec![Cell::new("..."); headers.len()]);
            table.add_row(ellipsis_row);
            println!("(Showing first {} of {} rows)", max_rows, dataset.data.len());
        }
        
        table.printstd();
    } else {
        println!("Dataset '{}' not found", name);
    }
}

/// List all available datasets
pub fn list_datasets() {
    let datasets = match DATASETS.read() {
        Ok(d) => d,
        Err(e) => {
            println!("Error listing datasets: {}", e);
            return;
        }
    };
    
    if datasets.is_empty() {
        println!("No datasets available.");
        return;
    }
    
    println!("Available datasets:");
    let mut table = Table::new();
    table.add_row(Row::new(vec![
        Cell::new("Name"),
        Cell::new("Rows"),
        Cell::new("Features"),
        Cell::new("Loaded"),
        Cell::new("Lazy")
    ]));
    
    for (name, ds) in datasets.iter() {
        let rows = if ds.loaded { ds.data.len().to_string() } else { "?".to_string() };
        let cols = if ds.loaded && !ds.data.is_empty() { ds.data[0].len().to_string() } else { "?".to_string() };
        
        table.add_row(Row::new(vec![
            Cell::new(name),
            Cell::new(&rows),
            Cell::new(&cols),
            Cell::new(if ds.loaded { "Yes" } else { "No" }),
            Cell::new(if ds.lazy { "Yes" } else { "No" })
        ]));
    }
    
    table.printstd();
}

/// Cleans up all datasets (useful for memory management or testing)
pub extern "C" fn luma_cleanup() {
    if let Ok(mut datasets) = DATASETS.write() {
        datasets.clear();
        println!("Cleared all datasets from memory.");
    } else {
        println!("Failed to acquire lock for dataset cleanup.");
    }
}

/// Load raw CSV data from a file
/// This is a lower-level function used by multi_modal.rs
pub fn load_csv(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let buf_reader = BufReader::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(buf_reader);

    let mut data = Vec::new();
    let mut row_count = 0;

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

    println!("Debug: Loaded {} rows into data (load_csv)", data.len());
    Ok(data)
}

/// Split dataset into training and testing sets
pub fn split_dataset(name: &str, test_ratio: f64) -> Result<(String, String), String> {
    // Get the dataset
    let dataset = match get_dataset(name) {
        Some(d) => d,
        None => return Err(format!("Dataset '{}' not found", name)),
    };
    
    if !dataset.loaded {
        return Err(format!("Dataset '{}' is not loaded", name));
    }
    
    if dataset.data.is_empty() {
        return Err(format!("Dataset '{}' is empty", name));
    }
    
    // Validate test ratio
    if test_ratio <= 0.0 || test_ratio >= 1.0 {
        return Err("Test ratio must be between 0 and 1 exclusive".to_string());
    }
    
    // Calculate split indices
    let total_rows = dataset.data.len();
    let test_size = (total_rows as f64 * test_ratio).round() as usize;
    let train_size = total_rows - test_size;
    
    if test_size == 0 || train_size == 0 {
        return Err(format!("Split would result in empty dataset (total rows: {})", total_rows));
    }
    
    // Create training dataset
    let train_name = format!("{}_train", name);
    let mut train_dataset = DatasetMetadata {
        name: train_name.clone(),
        path: None,
        headers: dataset.headers.clone(),
        data: dataset.data[0..train_size].to_vec(),
        labels: dataset.labels[0..train_size].to_vec(),
        lazy: false,
        loaded: true,
        label_column: dataset.label_column,
    };
    
    // Create testing dataset
    let test_name = format!("{}_test", name);
    let mut test_dataset = DatasetMetadata {
        name: test_name.clone(),
        path: None,
        headers: dataset.headers.clone(),
        data: dataset.data[train_size..].to_vec(),
        labels: dataset.labels[train_size..].to_vec(),
        lazy: false,
        loaded: true,
        label_column: dataset.label_column,
    };
    
    // Store the new datasets
    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    
    // Check for name conflicts
    if datasets.contains_key(&train_name) || datasets.contains_key(&test_name) {
        return Err(format!("Split dataset names ('{}' or '{}') already exist", train_name, test_name));
    }
    
    datasets.insert(train_name.clone(), train_dataset);
    datasets.insert(test_name.clone(), test_dataset);
    
    println!("Split dataset '{}' into '{}' ({} rows) and '{}' ({} rows)",
             name, train_name, train_size, test_name, test_size);
             
    Ok((train_name, test_name))
}

/// Clear all loaded datasets from memory
pub fn clear_datasets() {
    match DATASETS.write() {
        Ok(mut datasets) => {
            datasets.clear();
            println!("All datasets cleared from memory.");
        },
        Err(e) => {
            println!("Error clearing datasets: {}", e);
        }
    }
}