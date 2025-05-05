use std::sync::RwLock;
use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub struct Accelerator {}

lazy_static! {
    static ref ACCELERATOR: RwLock<Option<Accelerator>> = RwLock::new(None);
}

pub fn set_accelerator(_device: &str) -> Result<(), String> {
    let mut accel = ACCELERATOR.write().map_err(|_| "Failed to acquire write lock")?;
    *accel = Some(Accelerator {});
    Ok(())
}

pub fn get_accelerator() -> Option<Accelerator> {
    let accel = ACCELERATOR.read().map_err(|_| "Failed to acquire read lock").unwrap();
    accel.clone()
}

pub fn clear_accelerator() {
    let mut accel = ACCELERATOR.write().map_err(|_| "Failed to acquire write lock").unwrap();
    *accel = None;
}