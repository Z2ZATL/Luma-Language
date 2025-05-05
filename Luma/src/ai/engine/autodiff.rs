#[derive(Debug)]
pub struct Variable {
    value: f64,
    grad: f64,
}

impl Variable {
    pub fn new(value: f64) -> Self {
        Variable { value, grad: 0.0 }
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }
}

pub fn gradient_descent(v: &mut Variable, learning_rate: f64) {
    v.value -= learning_rate * v.grad;
}

#[no_mangle]
pub extern "C" fn luma_compute_gradient(value: f64, grad: f64) -> f64 {
    let mut var = Variable::new(value);
    var.set_grad(grad);
    gradient_descent(&mut var, 0.01); // Fixed learning rate for simplicity
    var.value
}