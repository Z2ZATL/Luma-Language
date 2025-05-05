use std::fs::OpenOptions;
use std::io::Write;

pub fn log_message(message: &str) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("luma.log")
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    writeln!(file, "[LOG] {}", message)
        .map_err(|e| format!("Failed to write to log file: {}", e))?;
    Ok(())
}

#[no_mangle]
pub extern "C" fn luma_log_message(message: *const std::os::raw::c_char) -> i32 {
    if message.is_null() {
        return -1;
    }
    let msg_str = unsafe { std::ffi::CStr::from_ptr(message).to_str() };
    if msg_str.is_err() {
        return -1;
    }
    match log_message(msg_str.unwrap()) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}