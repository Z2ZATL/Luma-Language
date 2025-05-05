#[cfg(test)]
mod tests {
    use crate::core::parser::{Lexer, Parser, ast::AstNode};
    use crate::core::compiler::codegen;

    #[test]
    fn test_generate_code_load() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        let code = codegen::generate_code(ast);
        assert_eq!(code, r#"luma_load_dataset("iris.csv", "iris", 1);"#);
    }

    #[test]
    fn test_generate_code_create() {
        let input = r#"create model neural_network"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        let code = codegen::generate_code(ast);
        assert_eq!(code, r#"luma_create_model("neural_network");"#);
    }
}