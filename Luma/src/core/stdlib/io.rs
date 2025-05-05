use std::fs::File;
use std::io::Write;

pub fn write_file(path: &str, content: &str) -> Result<(), String> {
    let mut file = File::create(path)
        .map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(())
}

#[no_mangle]
pub extern "C" fn luma_write_file(path: *const std::os::raw::c_char, content: *const std::os::raw::c_char) -> i32 {
    if path.is_null() || content.is_null() {
        return -1;
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str() };
    let content_str = unsafe { std::ffi::CStr::from_ptr(content).to_str() };
    if path_str.is_err() || content_str.is_err() {
        return -1;
    }
    match write_file(path_str.unwrap(), content_str.unwrap()) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}