#[derive(Debug)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err("Invalid reshape: size mismatch".to_string());
        }
        self.shape = new_shape;
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn luma_create_tensor(data: *const f64, len: i32, dims: i32, shape: *mut i32) -> i32 {
    if data.is_null() || shape.is_null() || len <= 0 || dims <= 0 {
        return -1;
    }
    unsafe {
        let data_slice = std::slice::from_raw_parts(data, len as usize);
        let mut shape_vec = Vec::new();
        for i in 0..dims as usize {
            shape_vec.push(*shape.offset(i as isize) as usize);
        }
        let tensor = Tensor::new(data_slice.to_vec(), shape_vec);
        // Placeholder: Store tensor in registry (to be implemented)
        0
    }
}