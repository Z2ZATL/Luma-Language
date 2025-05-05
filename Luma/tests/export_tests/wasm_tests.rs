#[cfg(test)]
mod tests {
    use crate::core::parser::{Lexer, Parser};
    use crate::core::compiler::ir;
    use crate::core::compiler::backend::wasm;

    #[test]
    fn test_compile_to_wasm() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        let ir = ir::ast_to_ir(ast);
        let code = wasm::compile_to_wasm(&ir);
        assert_eq!(code, r#"Module.load_dataset("iris.csv", "iris", true);"#);
    }
}