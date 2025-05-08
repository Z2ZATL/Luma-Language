#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    requires_grad: bool,
    grad: Option<Vec<f64>>,
    pub id: usize,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match the product of dimensions");

        Tensor { 
            data, 
            shape, 
            requires_grad: false, 
            grad: None,
            id: 0
        }
    }

    pub fn with_grad(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let mut tensor = Self::new(data, shape);
        tensor.requires_grad = true;
        tensor.grad = Some(vec![0.0; tensor.data.len()]);
        tensor
    }

    pub fn get_data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn get_grad(&self) -> Option<&Vec<f64>> {
        self.grad.as_ref()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(vec![0.0; self.data.len()]);
        }
    }

    pub fn zero_grad(&mut self) {
        if let Some(grad) = &mut self.grad {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
        }
    }

    pub fn accumulate_grad(&mut self, grad: &[f64]) {
        if self.requires_grad {
            if let Some(existing_grad) = &mut self.grad {
                assert_eq!(existing_grad.len(), grad.len(), "Gradient dimensions mismatch");
                for (i, &g) in grad.iter().enumerate() {
                    existing_grad[i] += g;
                }
            } else {
                self.grad = Some(grad.to_vec());
            }
        }
    }

    pub fn scale_grad(&mut self, factor: f64) {
        if let Some(grad) = &mut self.grad {
            for g in grad.iter_mut() {
                *g *= factor;
            }
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let total_size: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), total_size, "New shape must match the total number of elements");

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            id: self.id
        }
    }

    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut flat_index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            if indices[i] >= self.shape[i] {
                return None;
            }
            flat_index += indices[i] * stride;
            stride *= self.shape[i];
        }

        self.data.get(flat_index).copied()
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; total_size],
            shape,
            requires_grad: false,
            grad: None,
            id: 0
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; total_size],
            shape,
            requires_grad: false,
            grad: None,
            id: 0
        }
    }
}