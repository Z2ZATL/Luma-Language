use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::Reader;
use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
    data: Vec<Vec<f64>>, // Store loaded data as vectors
}

lazy_static! {
    static ref DATASETS: RwLock<Vec<DatasetMetadata>> = RwLock::new(Vec::new());
}

pub fn load_dataset(path: &str, name: &str) -> Result<(), String> {
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

    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.push(DatasetMetadata {
        name: name.to_string(),
        data,
    });
    Ok(())
}

pub fn get_dataset(name: &str) -> Option<DatasetMetadata> {
    let datasets = DATASETS.read().map_err(|e| format!("Lock error: {}", e)).ok()?;
    datasets.iter().find(|d| d.name == name).cloned()
}

pub fn print_dataset(name: &str) {
    if let Some(dataset) = get_dataset(name) {
        println!("Dataset Name: {}", dataset.name);
        println!("Data: {:?}", dataset.data);
    } else {
        println!("Dataset {} not found", name);
    }
}

pub extern "C" fn luma_cleanup() {
    let datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e)).ok();
    if let Some(mut datasets) = datasets {
        datasets.clear();
    }
}