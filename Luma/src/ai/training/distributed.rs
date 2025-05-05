use rayon::prelude::*;

pub fn distribute_training(data: &[f64]) -> Vec<f64> {
    data.par_iter().map(|x| *x * 2.0).collect() // Parallel computation
}

#[no_mangle]
pub extern "C" fn luma_distribute_training(data: *const f64, len: i32, out: *mut f64) -> i32 {
    if data.is_null() || out.is_null() || len <= 0 {
        return -1;
    }
    unsafe {
        let data_slice = std::slice::from_raw_parts(data, len as usize);
        let result = distribute_training(data_slice);
        std::ptr::copy_nonoverlapping(result.as_ptr(), out, len as usize);
    }
    0
}