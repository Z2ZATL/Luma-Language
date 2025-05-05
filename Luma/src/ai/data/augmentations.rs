use rand::Rng;
use std::sync::Arc;
use std::cell::RefCell;
use std::collections::HashMap;

// Assuming DatasetMetadata is defined in loaders.rs
#[derive(Debug)]
struct DatasetMetadata {
    id: i32,
    path: String,
    name: String,
    lazy: bool,
    data: Option<Vec<Vec<f64>>>,
    labels: Option<Vec<i32>>,
}

// Global dataset registry (shared with loaders.rs)
static mut DATASETS: Option<Arc<RefCell<HashMap<i32, DatasetMetadata>>>> = None;

// Initialize the dataset registry (should be called by luma_init)
pub fn initialize_data_registry() {
    unsafe {
        if DATASETS.is_none() {
            DATASETS = Some(Arc::new(RefCell::new(HashMap::new())));
        }
    }
}

// Augmentation configuration
#[derive(Debug)]
pub struct AugmentationConfig {
    rotate_prob: f64,  // Probability of applying rotation
    flip_prob: f64,    // Probability of applying horizontal flip
    noise_prob: f64,   // Probability of adding noise
    noise_scale: f64,  // Scale of Gaussian noise
}

impl AugmentationConfig {
    pub fn new(rotate_prob: f64, flip_prob: f64, noise_prob: f64, noise_scale: f64) -> Self {
        AugmentationConfig {
            rotate_prob,
            flip_prob,
            noise_prob,
            noise_scale,
        }
    }
}

// Apply augmentations to a dataset
pub fn apply_augmentations(dataset_id: i32, config: &AugmentationConfig) -> Result<(), String> {
    // Access the dataset
    let datasets = unsafe {
        DATASETS.as_ref()
            .ok_or("Data registry not initialized")?
    };
    let mut datasets_lock = datasets.borrow_mut();
    let dataset = datasets_lock.get_mut(&dataset_id)
        .ok_or(format!("Dataset with ID {} not found", dataset_id))?;

    // Ensure data is loaded
    if dataset.data.is_none() || dataset.labels.is_none() {
        return Err("Dataset not loaded".to_string());
    }

    let mut data = dataset.data.take().unwrap();
    let labels = dataset.labels.take().unwrap();

    // Apply augmentations
    let mut rng = rand::thread_rng();
    for i in 0..data.len() {
        // Rotate (simplified for numeric data: swap values in a row)
        if rng.gen::<f64>() < config.rotate_prob {
            let row = &mut data[i];
            row.reverse(); // Simplified rotation
        }

        // Flip (simplified: reverse order of features)
        if rng.gen::<f64>() < config.flip_prob {
            let row = &mut data[i];
            row.reverse();
        }

        // Add noise
        if rng.gen::<f64>() < config.noise_prob {
            let row = &mut data[i];
            for val in row.iter_mut() {
                let noise = rng.gen_range(-config.noise_scale..config.noise_scale);
                *val += noise;
            }
        }
    }

    // Update dataset
    dataset.data = Some(data);
    dataset.labels = Some(labels);

    Ok(())
}

// Clean up (shared with loaders.rs)
#[no_mangle]
pub extern "C" fn luma_cleanup() {
    unsafe {
        if let Some(datasets) = DATASETS.take() {
            let mut datasets_lock = datasets.borrow_mut();
            datasets_lock.clear();
        }
    }
}