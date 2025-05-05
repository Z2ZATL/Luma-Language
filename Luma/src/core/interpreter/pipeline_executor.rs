use crate::parser::lexer::Lexer;
use crate::parser::parser::Parser;

pub fn execute_pipeline(input: &str) -> Result<(), String> {
    let lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;
    // Placeholder for execution logic
    println!("AST: {:?}", ast);
    Ok(())
}

#[no_mangle]
pub extern "C" fn luma_execute_pipeline(input: *const std::os::raw::c_char) -> i32 {
    if input.is_null() {
        return -1;
    }
    let input_str = unsafe { std::ffi::CStr::from_ptr(input).to_str() };
    if input_str.is_err() {
        return -1;
    }
    match execute_pipeline(input_str.unwrap()) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}