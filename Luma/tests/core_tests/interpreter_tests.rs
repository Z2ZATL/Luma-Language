#[cfg(test)]
mod tests {
    use crate::core::interpreter::pipeline_executor;

    #[test]
    fn test_execute_pipeline_load() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let result = pipeline_executor::execute_pipeline(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_pipeline_create() {
        let input = r#"create model neural_network"#;
        let result = pipeline_executor::execute_pipeline(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_input() {
        let input = r#"invalid command"#;
        let result = pipeline_executor::execute_pipeline(input);
        assert!(result.is_err());
    }
}