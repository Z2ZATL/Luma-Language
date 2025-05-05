use crate::core_parser::lexer::Lexer;
use crate::core_parser::parser::Parser;
use crate::core_interpreter::evaluator;
use crate::core_compiler::ir;
use crate::core_compiler::codegen;

pub fn execute_pipeline(input: &str) -> Result<(), String> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;
    let ir = ir::generate_ir(ast.clone());
    let code = codegen::generate_code(ast.clone());
    println!("Generated IR: {:?}", ir);
    println!("Generated Code: {}", code);
    let mut scope = crate::core_parser::ast::Scope::new();
    evaluator::evaluate(ast, &mut scope)?;
    Ok(())
}