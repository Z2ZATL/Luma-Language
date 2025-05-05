use std::sync::RwLock;
use lazy_static::lazy_static;

lazy_static! {
    static ref OPTIMIZED_MODELS: RwLock<Vec<String>> = RwLock::new(Vec::new());
}

/// Optimizes a model by adding it to the registry
pub fn optimize_model(model_id: i32) -> Result<(), String> {
    let mut registry = OPTIMIZED_MODELS.write().map_err(|_| "Failed to acquire write lock")?;
    registry.push(format!("Model_{}", model_id));
    Ok(())
}

/// Clears all optimized models from the registry
pub fn clear_optimized_models() -> Result<(), String> {
    let mut registry = OPTIMIZED_MODELS.write().map_err(|_| "Failed to acquire write lock")?;
    registry.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_model() {
        assert!(optimize_model(1).is_ok());
        assert!(optimize_model(2).is_ok());
    }

    #[test]
    fn test_clear_optimized_models() {
        assert!(clear_optimized_models().is_ok());
    }
}