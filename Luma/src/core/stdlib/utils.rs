use std::os::raw::c_char;
use std::ffi::CString;

pub fn to_string(x: f64) -> String {
    x.to_string()
}

#[no_mangle]
pub extern "C" fn luma_to_string(x: f64) -> *mut c_char {
    let s = to_string(x);
    let c_str = CString::new(s).unwrap();
    c_str.into_raw()
}

pub fn normalize_data(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let row_count = data.len();
    if row_count == 0 {
        return data.clone();
    }
    let col_count = data[0].len();
    let mut normalized = Vec::with_capacity(row_count);
    for row in data {
        let mut normalized_row = Vec::with_capacity(col_count);
        for &val in row {
            normalized_row.push(val / 255.0); // Example normalization
        }
        normalized.push(normalized_row);
    }
    normalized
}

pub fn scale_data(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let row_count = data.len();
    if row_count == 0 {
        return data.clone();
    }
    let col_count = data[0].len();
    let mut scaled = Vec::with_capacity(row_count);
    for row in data {
        let mut scaled_row = Vec::with_capacity(col_count);
        for &val in row {
            scaled_row.push(val * 2.0); // Example scaling
        }
        scaled.push(scaled_row);
    }
    scaled
}