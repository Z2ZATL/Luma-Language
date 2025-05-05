#[cfg(test)]
mod tests {
    use crate::ai::deployment::deployers;
    use std::ffi::CString;

    #[test]
    fn test_deploy_model() {
        let target = CString::new("server").unwrap();
        let output_path = CString::new("model.deployment").unwrap();
        let result = unsafe {
            deployers::luma_deploy_model(1, target.as_ptr(), output_path.as_ptr())
        };
        assert_eq!(result, 0);
    }
}