use std::sync::RwLock;
use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub struct MultiModalData {}

lazy_static! {
    static ref MULTI_MODAL_REGISTRY: RwLock<Vec<MultiModalData>> = RwLock::new(Vec::new());
}

pub fn register_multi_modal(_path: &str) -> Result<(), String> {
    let mut registry = MULTI_MODAL_REGISTRY.write().map_err(|_| "Failed to acquire write lock")?;
    registry.push(MultiModalData {});
    Ok(())
}

pub fn get_multi_modal() -> Option<Vec<MultiModalData>> {
    let registry = MULTI_MODAL_REGISTRY.read().map_err(|_| "Failed to acquire read lock").unwrap();
    Some(registry.clone())
}

pub fn clear_multi_modal() {
    let mut registry = MULTI_MODAL_REGISTRY.write().map_err(|_| "Failed to acquire write lock").unwrap();
    registry.clear();
}