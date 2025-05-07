
use crate::ai::engine::tensor::Tensor;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct Operation {
    op_type: String,
    inputs: Vec<usize>,   // Tensor IDs
    output: usize,        // Tensor ID
    metadata: HashMap<String, String>, // For storing extra info like scalar values
}

#[derive(Debug)]
pub struct ComputationGraph {
    operations: Vec<Operation>,
    tensors: HashMap<usize, Tensor>,
    next_id: usize,
}

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

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            operations: Vec::new(),
            tensors: HashMap::new(),
            next_id: 1, // Start IDs at 1
        }
    }
    
    pub fn register_tensor(&mut self, tensor: Tensor) -> Tensor {
        let mut tensor = tensor.clone();
        let id = self.next_id;
        self.next_id += 1;
        
        // Set the ID in the tensor
        tensor.id = id;
        
        // Store the tensor
        self.tensors.insert(id, tensor.clone());
        
        tensor
    }
    
    pub fn add_operation(&mut self, op_type: &str, inputs: Vec<Tensor>, output: Tensor) -> usize {
        // Register input tensors if they're not already in the graph
        let input_ids: Vec<usize> = inputs.iter()
            .map(|t| {
                if t.id == 0 {
                    self.register_tensor(t.clone()).id
                } else {
                    t.id
                }
            })
            .collect();
        
        // Register output tensor
        let output_id = if output.id == 0 {
            self.register_tensor(output).id
        } else {
            output.id
        };
        
        // Create and add the operation
        let op_id = self.operations.len();
        self.operations.push(Operation {
            op_type: op_type.to_string(),
            inputs: input_ids,
            output: output_id,
            metadata: HashMap::new(),
        });
        
        op_id
    }
    
    pub fn set_op_metadata(&mut self, op_id: usize, key: &str, value: String) {
        if op_id < self.operations.len() {
            self.operations[op_id].metadata.insert(key.to_string(), value);
        }
    }
    
    pub fn backward(&mut self, output_id: usize) {
        // Initialize gradient of output to 1.0
        if let Some(output) = self.tensors.get_mut(&output_id) {
            let grad_data = vec![1.0; output.get_data().len()];
            output.accumulate_grad(&grad_data);
        } else {
            panic!("Output tensor not found in graph");
        }
        
        // Topologically sort operations for backward pass
        let mut op_queue: VecDeque<usize> = VecDeque::new();
        
        // Find operations that output to the target tensor
        for (i, op) in self.operations.iter().enumerate() {
            if op.output == output_id {
                op_queue.push_back(i);
            }
        }
        
        // Process operations in reverse order
        while let Some(op_idx) = op_queue.pop_front() {
            let op = &self.operations[op_idx];
            
            // Get the output tensor and its gradient
            let output_tensor = self.tensors.get(&op.output).unwrap().clone();
            let output_grad = output_tensor.get_grad().unwrap().clone();
            
            match op.op_type.as_str() {
                "add" => {
                    // Gradient flows unchanged to both inputs
                    for &input_id in &op.inputs {
                        if let Some(input_tensor) = self.tensors.get_mut(&input_id) {
                            if input_tensor.requires_grad() {
                                input_tensor.accumulate_grad(&output_grad);
                            }
                        }
                    }
                },
                "mul" => {
                    // For element-wise multiplication: dz/dx = y * dz/dz and dz/dy = x * dz/dz
                    let x = self.tensors.get(&op.inputs[0]).unwrap();
                    let y = self.tensors.get(&op.inputs[1]).unwrap();
                    
                    // Calculate gradients for each input
                    if x.requires_grad() {
                        let dx: Vec<f64> = y.get_data().iter()
                            .zip(output_grad.iter())
                            .map(|(&y_val, &grad)| y_val * grad)
                            .collect();
                        
                        if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[0]) {
                            input_tensor.accumulate_grad(&dx);
                        }
                    }
                    
                    if y.requires_grad() {
                        let dy: Vec<f64> = x.get_data().iter()
                            .zip(output_grad.iter())
                            .map(|(&x_val, &grad)| x_val * grad)
                            .collect();
                        
                        if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[1]) {
                            input_tensor.accumulate_grad(&dy);
                        }
                    }
                },
                "matmul" => {
                    // For matrix multiplication: dz/dx = dz/dz @ y.T and dz/dy = x.T @ dz/dz
                    let x = self.tensors.get(&op.inputs[0]).unwrap();
                    let y = self.tensors.get(&op.inputs[1]).unwrap();
                    
                    let x_shape = x.get_shape();
                    let y_shape = y.get_shape();
                    
                    // Gradient computation for matmul is complex and would be implemented here
                    // This is a simplified placeholder
                    if x.requires_grad() {
                        // Simplified gradient calculation for X
                        let mut dx = vec![0.0; x.get_data().len()];
                        // Actual implementation would compute X gradient from Y and output gradient
                        
                        if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[0]) {
                            input_tensor.accumulate_grad(&dx);
                        }
                    }
                    
                    if y.requires_grad() {
                        // Simplified gradient calculation for Y
                        let mut dy = vec![0.0; y.get_data().len()];
                        // Actual implementation would compute Y gradient from X and output gradient
                        
                        if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[1]) {
                            input_tensor.accumulate_grad(&dy);
                        }
                    }
                },
                "sum" | "sum_all" => {
                    // For sum, gradient is 1.0 for each element
                    if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[0]) {
                        if input_tensor.requires_grad() {
                            // For sum, we broadcast the gradient
                            let mut dx = vec![1.0; input_tensor.get_data().len()];
                            
                            // Scale by the output gradient
                            for i in 0..dx.len() {
                                dx[i] *= output_grad[0]; // Assuming scalar output for now
                            }
                            
                            input_tensor.accumulate_grad(&dx);
                        }
                    }
                },
                "scalar_mul" => {
                    // For scalar multiplication: dz/dx = scalar * dz/dz
                    if let Some(scalar_str) = op.metadata.get("scalar") {
                        if let Ok(scalar) = scalar_str.parse::<f64>() {
                            if let Some(input_tensor) = self.tensors.get_mut(&op.inputs[0]) {
                                if input_tensor.requires_grad() {
                                    let dx: Vec<f64> = output_grad.iter()
                                        .map(|&grad| scalar * grad)
                                        .collect();
                                    
                                    input_tensor.accumulate_grad(&dx);
                                }
                            }
                        }
                    }
                },
                _ => {
                    panic!("Unsupported operation type in backward pass: {}", op.op_type);
                }
            }
            
            // Add operations that output to the inputs to the queue
            for &input_id in &op.inputs {
                for (i, op) in self.operations.iter().enumerate() {
                    if op.output == input_id {
                        op_queue.push_back(i);
                    }
                }
            }
        }
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
