use std::sync::RwLock;
use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    name: String,
}

lazy_static! {
    static ref DATASETS: RwLock<Vec<DatasetMetadata>> = RwLock::new(Vec::new());
}

pub fn load_dataset(_path: &str, name: &str) -> Result<(), String> {
    let mut datasets = DATASETS.write().map_err(|_| "Failed to acquire write lock")?;
    datasets.push(DatasetMetadata {
        name: name.to_string(),
    });
    Ok(())
}

pub fn get_dataset(name: &str) -> Option<DatasetMetadata> {
    let datasets = DATASETS.read().map_err(|_| "Failed to acquire read lock").unwrap();
    datasets.iter().find(|d| d.name == name).cloned()
}

pub extern "C" fn luma_cleanup() {
    let mut datasets = DATASETS.write().map_err(|_| "Failed to acquire write lock").unwrap();
    datasets.clear();
}