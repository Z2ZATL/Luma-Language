#[cfg(test)]
mod tests {
    use crate::integrations::tensorflow;
    use std::ffi::CString;

    #[test]
    fn test_load_tensorflow_model() {
        let model_path = CString::new("model.tf").unwrap();
        let result = unsafe { tensorflow::luma_load_tensorflow_model(model_path.as_ptr()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_invalid_path() {
        let model_path = CString::new("").unwrap();
        let result = unsafe { tensorflow::luma_load_tensorflow_model(model_path.as_ptr()) };
        assert_eq!(result, -1);
    }
}