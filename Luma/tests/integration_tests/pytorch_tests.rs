#[cfg(test)]
mod tests {
    use crate::integrations::pytorch;
    use std::ffi::CString;

    #[test]
    fn test_load_pytorch_model() {
        let model_path = CString::new("model.pt").unwrap();
        let result = unsafe { pytorch::luma_load_pytorch_model(model_path.as_ptr()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_invalid_path() {
        let model_path = CString::new("").unwrap();
        let result = unsafe { pytorch::luma_load_pytorch_model(model_path.as_ptr()) };
        assert_eq!(result, -1);
    }
}