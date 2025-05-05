use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::cell::RefCell;
use std::collections::HashMap;

// Structure to hold dataset metadata
#[derive(Debug)]
struct DatasetMetadata {
    id: i32,
    path: String,
    name: String,
    lazy: bool,
    data: Option<Vec<Vec<f64>>>, // Lazy-loaded data
    labels: Option<Vec<i32>>,
}

impl DatasetMetadata {
    fn new(id: i32, path: String, name: String, lazy: bool) -> Self {
        DatasetMetadata {
            id,
            path,
            name,
            lazy,
            data: if !lazy { Some(Vec::new()) } else { None },
            labels: if !lazy { Some(Vec::new()) } else { None },
        }
    }

    fn load(&mut self) -> io::Result<()> {
        if self.lazy || self.data.is_some() {
            return Ok(());
        }
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let values: Vec<&str> = line.split(',').collect();
            let mut row = Vec::new();
            for val in values[..values.len() - 1].iter() {
                row.push(val.parse::<f64>().unwrap_or_else(|_| panic!("Invalid numeric value")));
            }
            let label = values[values.len() - 1].parse::<i32>().unwrap_or_else(|_| panic!("Invalid label value"));
            self.data.as_mut().unwrap().push(row);
            self.labels.as_mut().unwrap().push(label);
        }
        Ok(())
    }
}

// Global dataset registry
static mut DATASETS: Option<Arc<RefCell<HashMap<i32, DatasetMetadata>>>> = None;

pub fn initialize_data_registry() {
    unsafe {
        if DATASETS.is_none() {
            DATASETS = Some(Arc::new(RefCell::new(HashMap::new())));
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_load_dataset(path: *const std::os::raw::c_char, name: *const std::os::raw::c_char, lazy: std::os::raw::c_int) -> std::os::raw::c_int {
    if path.is_null() || name.is_null() {
        return -1;
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str() };
    let name_str = unsafe { std::ffi::CStr::from_ptr(name).to_str() };
    if path_str.is_err() || name_str.is_err() {
        return -1;
    }
    let path = path_str.unwrap().to_string();
    let name = name_str.unwrap().to_string();
    let lazy_flag = lazy != 0;

    unsafe {
        let datasets = DATASETS.as_ref().expect("Data registry not initialized");
        let mut datasets_lock = datasets.borrow_mut();
        let id = datasets_lock.len() as i32 + 1;
        let mut metadata = DatasetMetadata::new(id, path, name, lazy_flag);
        if !lazy_flag {
            if let Err(e) = metadata.load() {
                eprintln!("Failed to load dataset: {}", e);
                return -1;
            }
        }
        datasets_lock.insert(id, metadata);
        id
    }
}

#[no_mangle]
pub extern "C" fn luma_cleanup() {
    unsafe {
        if let Some(datasets) = DATASETS.take() {
            let mut datasets_lock = datasets.borrow_mut();
            datasets_lock.clear();
        }
    }
}