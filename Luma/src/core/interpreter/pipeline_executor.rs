use super::evaluator;
use super::super::parser::{Lexer, Parser};

pub fn execute_pipeline(input: &str) -> Result<(), String> {
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let ast = parser.parse()?;
    evaluator::evaluate(ast)
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