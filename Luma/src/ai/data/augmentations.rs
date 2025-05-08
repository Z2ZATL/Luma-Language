use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::ReaderBuilder;
use lazy_static::lazy_static;
use std::collections::HashMap;
use prettytable::{Table, Row, Cell};
use crate::core::stdlib::utils::{normalize_data, scale_data}; // Import from utils.rs

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
    path: String, // Store the path for lazy loading
    headers: Option<Vec<String>>, // Store headers if present
    data: Option<Vec<Vec<f64>>>, // Lazy-loaded data
}

lazy_static! {
    static ref DATASETS: RwLock<HashMap<String, DatasetMetadata>> = RwLock::new(HashMap::new());
}

pub fn augment_dataset(path: &str, name: &str, lazy: bool) -> Result<(), String> {
    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    // Check for duplicate name
    if datasets.contains_key(name) {
        return Err(format!("Augmentation name '{}' already exists", name));
    }

    let (headers, data) = if !lazy {
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
        let normalized = normalize_data(&data);
        let scaled = scale_data(&normalized);
        (headers, Some(scaled))
    } else {
        (None, None) // Lazy loading will handle headers and data later
    };

    datasets.insert(
        name.to_string(),
        DatasetMetadata {
            name: name.to_string(),
            path: path.to_string(),
            headers,
            data,
        },
    );
    Ok(())
}

pub fn get_augmentation(name: &str) -> Option<DatasetMetadata> {
    let datasets = DATASETS.read().map_err(|e| format!("Lock error: {}", e)).ok()?;
    let mut dataset = datasets.get(name)?.clone();
    if dataset.data.is_none() {
        // Lazy loading logic using the stored path
        let file = File::open(&dataset.path).map_err(|e| format!("Failed to open file: {}", e)).ok()?;
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
            let record = result.map_err(|e| format!("Failed to read CSV at row {}: {}", row_count, e)).ok()?;
            let row: Vec<f64> = record.iter()
                .filter_map(|field| field.parse::<f64>().ok())
                .collect();
            if !row.is_empty() {
                data.push(row);
            } else {
                println!("Debug: Empty row detected at row {} (get_augmentation)", row_count);
            }
        }
        if !data.is_empty() {
            let normalized = normalize_data(&data);
            let scaled = scale_data(&normalized);
            dataset.data = Some(scaled);
            dataset.headers = headers;
            println!("Debug: Loaded {} rows into data (get_augmentation)", data.len());
        }
    }
    Some(dataset)
}

pub fn print_augmentation(name: &str) {
    if let Some(dataset) = get_augmentation(name) {
        println!("Augmentation Name: {}", dataset.name);
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
            // Add data rows
            for row in data {
                let cells: Vec<Cell> = row.iter()
                    .map(|val| Cell::new(&format!("{:.3}", val)))
                    .collect();
                table.add_row(Row::new(cells));
            }
            table.printstd();
        } else {
            println!("Data: None");
        }
    } else {
        println!("Augmentation {} not found", name);
    }
}

pub fn clear_augmentations() {
    let datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e)).ok();
    if let Some(mut datasets) = datasets {
        datasets.clear();
    }
}