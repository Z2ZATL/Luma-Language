#[cfg(test)]
mod tests {
    use crate::core::parser::{Lexer, Parser, ast::AstNode};

    #[test]
    fn test_load_dataset() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        assert_eq!(ast, AstNode::LoadDataset {
            path: "iris.csv".to_string(),
            name: "iris".to_string(),
            lazy: true,
        });
    }

    #[test]
    fn test_create_model() {
        let input = r#"create model neural_network"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        assert_eq!(ast, AstNode::CreateModel {
            model_type: "neural_network".to_string(),
        });
    }

    #[test]
    fn test_invalid_syntax() {
        let input = r#"load dataset "iris.csv""#; // Missing 'as'
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse();
        assert!(result.is_err());
    }
}