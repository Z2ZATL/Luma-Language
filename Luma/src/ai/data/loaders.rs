use std::fs::File;
use std::io::BufReader;
use std::sync::RwLock;
use csv::ReaderBuilder;
use lazy_static::lazy_static;
use prettytable::{Table, Row, Cell};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
    headers: Option<Vec<String>>,
    data: Vec<Vec<f64>>, // Features only
    labels: Vec<Vec<f64>>, // Labels from the last column (as Vec<f64> for compatibility)
}

impl DatasetMetadata {
    pub fn get_data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    pub fn get_labels(&self) -> &Vec<Vec<f64>> {
        &self.labels
    }
}

lazy_static! {
    static ref DATASETS: RwLock<Vec<DatasetMetadata>> = RwLock::new(Vec::new());
}

pub fn load_dataset(path: &str, name: &str) -> Result<(), String> {
    let datasets = DATASETS.read().map_err(|e| format!("Lock error: {}", e))?;
    if datasets.iter().any(|d| d.name == name) {
        return Err(format!("Dataset name '{}' already exists", name));
    }
    drop(datasets);

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let buf_reader = BufReader::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(buf_reader);

    let mut data = Vec::new();
    let mut labels = Vec::new();
    let mut headers = None;
    let mut row_count = 0;

    match reader.headers() {
        Ok(header_record) => {
            headers = Some(header_record.iter().map(|h| h.to_string()).collect());
        }
        Err(e) => println!("Warning: Failed to read headers: {}. Proceeding without headers.", e),
    }

    for result in reader.records() {
        row_count += 1;
        let record = result.map_err(|e| format!("Failed to read CSV at row {}: {}", row_count, e))?;
        let values: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        if values.len() < 2 {
            println!("Debug: Skipping row {} due to insufficient columns", row_count);
            continue;
        }
        let row: Vec<f64> = values[..values.len() - 1]
            .iter()
            .filter_map(|field| field.parse::<f64>().ok())
            .collect();
        let label: f64 = values[values.len() - 1].parse().unwrap_or(0.0); // Default to 0 if parsing fails
        if !row.is_empty() {
            data.push(row);
            labels.push(vec![label]); // Wrap label in a Vec for compatibility with trainers.rs
        } else {
            println!("Debug: Empty row detected at row {}", row_count);
        }
    }

    if data.is_empty() {
        return Err(format!("No valid data found in the CSV file. Processed {} rows.", row_count));
    }

    println!("Debug: Loaded {} rows into data", data.len());

    let mut datasets = DATASETS.write().map_err(|e| format!("Lock error: {}", e))?;
    datasets.push(DatasetMetadata {
        name: name.to_string(),
        headers,
        data,
        labels,
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
        let mut table = Table::new();
        let headers: Vec<Cell> = match &dataset.headers {
            Some(h) => h.iter().map(|h| Cell::new(h)).collect(),
            None => (0..dataset.data[0].len() + 1)
                .map(|i| Cell::new(&format!("Col {}", i)))
                .collect(),
        };
        table.add_row(Row::new(headers));
        for (i, row) in dataset.data.iter().enumerate() {
            let mut cells: Vec<Cell> = row.iter()
                .map(|val| Cell::new(&format!("{:.1}", val)))
                .collect();
            cells.push(Cell::new(&format!("{:.1}", dataset.labels[i][0])));
            table.add_row(Row::new(cells));
        }
        table.printstd();
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