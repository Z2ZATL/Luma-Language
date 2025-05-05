use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::Reader;
use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
    data: Option<Vec<Vec<f64>>>, // Lazy-loaded data
}

lazy_static! {
    static ref DATASETS: RwLock<HashMap<String, DatasetMetadata>> = RwLock::new(HashMap::new());
}

pub fn augment_dataset(path: &str, name: &str, lazy: bool) -> Result<(), String> {
    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    let data = if !lazy {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);
        let mut data = Vec::new();
        for result in reader.records() {
            let record = result.map_err(|e| format!("Failed to read CSV: {}", e))?;
            let row: Vec<f64> = record.iter()
                .filter_map(|field| field.parse::<f64>().ok())
                .collect();
            data.push(row);
        }
        Some(normalize_data(&data))
    } else {
        None // Lazy loading will load data when accessed
    };

    datasets.insert(
        name.to_string(),
        DatasetMetadata { name: name.to_string(), data },
    );
    Ok(())
}

fn normalize_data(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let row_count = data.len();
    if row_count == 0 {
        return data.clone();
    }
    let col_count = data[0].len();
    let mut normalized = Vec::with_capacity(row_count);
    for row in data {
        let mut normalized_row = Vec::with_capacity(col_count);
        for &val in row {
            normalized_row.push(val / 255.0); // Simple normalization (e.g., for image data)
        }
        normalized.push(normalized_row);
    }
    normalized
}

pub fn get_augmentation(name: &str) -> Option<DatasetMetadata> {
    let datasets = DATASETS.read().map_err(|e| format!("Lock error: {}", e)).ok()?;
    let mut dataset = datasets.get(name)?.clone();
    if dataset.data.is_none() {
        // Lazy loading logic
        let file = File::open("dummy_path.csv").map_err(|e| format!("Failed to open file: {}", e)).ok(); // Replace with actual path logic
        if let Some(file) = file {
            let buf_reader = BufReader::new(file);
            let mut reader = Reader::from_reader(buf_reader);
            let mut data = Vec::new();
            for result in reader.records() {
                let record = result.map_err(|e| format!("Failed to read CSV: {}", e)).ok()?;
                let row: Vec<f64> = record.iter()
                    .filter_map(|field| field.parse::<f64>().ok())
                    .collect();
                data.push(row);
            }
            dataset.data = Some(normalize_data(&data));
        }
    }
    Some(dataset)
}

pub fn print_augmentation(name: &str) {
    if let Some(dataset) = get_augmentation(name) {
        println!("Augmentation Name: {}", dataset.name);
        println!("Data: {:?}", dataset.data);
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