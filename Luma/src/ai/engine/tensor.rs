pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        let _tensor = Tensor { data: data.clone(), shape: shape.clone() };
        Tensor { data, shape }
    }

    pub fn get_data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }
}