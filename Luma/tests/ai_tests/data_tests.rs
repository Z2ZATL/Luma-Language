#[cfg(test)]
mod tests {
    use crate::ai::data::loaders;
    use crate::ai::data::augmentations::{AugmentationConfig, apply_augmentations};
    use std::ffi::CString;

    #[test]
    fn test_load_dataset() {
        loaders::initialize_data_registry();
        let path = CString::new("iris.csv").unwrap();
        let name = CString::new("iris").unwrap();
        let result = unsafe { loaders::luma_load_dataset(path.as_ptr(), name.as_ptr(), 1) };
        assert!(result > 0);
    }

    #[test]
    fn test_augmentation() {
        loaders::initialize_data_registry();
        let path = CString::new("iris.csv").unwrap();
        let name = CString::new("iris").unwrap();
        let dataset_id = unsafe { loaders::luma_load_dataset(path.as_ptr(), name.as_ptr(), 0) };
        let config = AugmentationConfig::new(0.5, 0.5, 0.5, 0.1);
        let result = apply_augmentations(dataset_id, &config);
        assert!(result.is_ok());
    }
}