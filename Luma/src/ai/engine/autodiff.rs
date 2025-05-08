use crate::ai::engine::tensor::Tensor;
use std::collections::{HashMap, VecDeque};
use crate::utilities::debugging::set_debug_level;
use crate::debug_print;

#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: String,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ComputationGraph {
    pub operations: Vec<Operation>,
    pub tensors: HashMap<usize, Tensor>,
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
            next_id: 1,
        }
    }

    pub fn register_tensor(&mut self, tensor: Tensor) -> Tensor {
        let mut tensor = tensor.clone();
        let id = self.next_id;
        self.next_id += 1;

        tensor.id = id;
        self.tensors.insert(id, tensor.clone());

        tensor
    }

    pub fn add_operation(&mut self, op_type: &str, inputs: Vec<Tensor>, output: Tensor) -> usize {
        let input_ids: Vec<usize> = inputs.iter()
            .map(|t| {
                if t.id == 0 {
                    let registered = self.register_tensor(t.clone());
                    debug_print!(2, "Debug: Registered input tensor for op {} with ID {}", op_type, registered.id);
                    registered.id
                } else {
                    debug_print!(3, "Debug: Input tensor for op {} already registered with ID {}", op_type, t.id);
                    t.id
                }
            })
            .collect();

        let output_id = if output.id == 0 {
            let registered = self.register_tensor(output);
            debug_print!(2, "Debug: Registered output tensor for op {} with ID {}", op_type, registered.id);
            registered.id
        } else {
            debug_print!(3, "Debug: Output tensor for op {} already registered with ID {}", op_type, output.id);
            output.id
        };

        let op_id = self.operations.len();
        self.operations.push(Operation {
            op_type: op_type.to_string(),
            inputs: input_ids.clone(),
            output: output_id,
            metadata: HashMap::new(),
        });
        debug_print!(1, "Debug: Added operation {} (type: {}), inputs: {:?}, output: {}", op_id, op_type, input_ids, output_id);

        op_id
    }

    pub fn set_op_metadata(&mut self, op_id: usize, key: &str, value: String) {
        if op_id < self.operations.len() {
            self.operations[op_id].metadata.insert(key.to_string(), value);
        }
    }

    pub fn backward(&mut self, output_id: usize) {
        debug_print!(1, "Debug: Starting backward pass from tensor {}", output_id);
        
        // First, check if we have any operations that actually produce this tensor
        let mut has_producing_ops = false;
        for (op_id, op) in self.operations.iter().enumerate() {
            if op.output == output_id {
                has_producing_ops = true;
                debug_print!(2, "Debug: Found op {} (type: {}) that produces tensor {}", op_id, op.op_type, output_id);
            }
        }
        
        if !has_producing_ops {
            debug_print!(1, "Debug: WARNING - No operations found producing tensor {}. This may disconnect the computational graph.", output_id);
        }
        
        // Initialize gradient of the output tensor
        if let Some(tensor) = self.tensors.get_mut(&output_id) {
            // Create ones gradient vector with same size as tensor data
            let ones = vec![1.0; tensor.get_data().len()];
            
            // Initialize gradient if the tensor requires gradients
            if tensor.requires_grad() {
                // Manually accumulate gradient
                if let Some(existing_grad) = tensor.get_grad() {
                    // Deep clone to work with
                    let mut new_grad = existing_grad.to_vec();
                    for i in 0..new_grad.len() {
                        new_grad[i] = 1.0;
                    }
                    // We need to update the tensor with this new gradient
                    tensor.accumulate_grad(&ones);
                } else {
                    // If no gradient exists yet, just accumulate ones
                    tensor.accumulate_grad(&ones);
                }
                debug_print!(2, "Debug: Initialized gradient for output tensor {}: {:?}", output_id, tensor.get_grad());
            } else {
                debug_print!(1, "Debug: WARNING - Output tensor {} does not require gradients", output_id);
            }
        } else {
            debug_print!(1, "Debug: Error - Output tensor {} not found", output_id);
            return;
        }
        
        // Print computational graph summary for debugging
        debug_print!(2, "Debug: Computational graph summary:");
        debug_print!(2, "Debug: Total operations: {}", self.operations.len());
        debug_print!(2, "Debug: Total tensors: {}", self.tensors.len());
        
        // Print a few operations to check connectivity
        let limit = 5.min(self.operations.len());
        for op_idx in 0..limit {
            let op = &self.operations[self.operations.len() - 1 - op_idx];
            debug_print!(3, "Debug: Recent op {}: type={}, inputs={:?}, output={}", 
                     self.operations.len() - 1 - op_idx, op.op_type, op.inputs, op.output);
        }

        // Create a queue of operations to process
        let mut op_queue = VecDeque::new();

        // Create a set to track visited operations (prevent duplicates)
        let mut visited_ops = std::collections::HashSet::new();

        // Find operations that produce the output tensor
        for (op_id, op) in self.operations.iter().enumerate() {
            if op.output == output_id {
                op_queue.push_back(op_id);
                visited_ops.insert(op_id);
                println!("Debug: Added op {} to queue (output matches tensor {})", op_id, output_id);
            }
        }

        // Process operations in topological order
        while let Some(op_id) = op_queue.pop_front() {
            let op = match self.operations.get(op_id) {
                Some(op) => op.clone(),
                None => {
                    println!("Debug: Warning - Operation {} not found", op_id);
                    continue;
                }
            };

            // Get output gradient
            let output_grad = match self.tensors.get(&op.output) {
                Some(tensor) => tensor.get_grad(),
                None => {
                    println!("Debug: Warning - Tensor {} not found", op.output);
                    continue;
                }
            };

            println!("Debug: Processing op {} (type: {}), output grad: {:?}", op_id, op.op_type, output_grad);

            // Compute gradient for input tensors based on operation type
            match op.op_type.as_str() {
                "matmul" => {
                    // For matrix multiplication: if z = x * y, then dz/dx = y^T, dz/dy = x^T
                    if op.inputs.len() >= 2 {
                        let input_a_id = op.inputs[0];
                        let input_b_id = op.inputs[1];
                        
                        // Clone necessary data to avoid borrow checker issues
                        let (output_grad_clone, input_a_data_clone, input_b_data_clone) = {
                            if let (Some(input_a), Some(input_b), Some(output_tensor)) = (
                                self.tensors.get(&input_a_id), 
                                self.tensors.get(&input_b_id),
                                self.tensors.get(&op.output)
                            ) {
                                if let Some(output_grad) = output_tensor.get_grad() {
                                    (
                                        output_grad.clone(),
                                        input_a.get_data().clone(),
                                        input_b.get_data().clone()
                                    )
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        // Compute gradients for input_a (first operand)
                        if let Some(input_a) = self.tensors.get(&input_a_id) {
                            if input_a.requires_grad() {
                                let mut grad_a = vec![0.0; input_a_data_clone.len()];
                                
                                // Simple case: grad_a = output_grad * input_b
                                for i in 0..grad_a.len() {
                                    if i < input_b_data_clone.len() {
                                        grad_a[i] = output_grad_clone[0] * input_b_data_clone[i];
                                    }
                                }
                                
                                if let Some(input_a_mut) = self.tensors.get_mut(&input_a_id) {
                                    input_a_mut.accumulate_grad(&grad_a);
                                    println!("Debug: Matmul - Updated grad for input_a tensor {}: {:?}", input_a_id, input_a_mut.get_grad());
                                }
                            }
                        }
                        
                        // Compute gradients for input_b (second operand)
                        if let Some(input_b) = self.tensors.get(&input_b_id) {
                            if input_b.requires_grad() {
                                let mut grad_b = vec![0.0; input_b_data_clone.len()];
                                
                                // Simple case: grad_b = output_grad * input_a
                                for i in 0..grad_b.len() {
                                    if i < input_a_data_clone.len() {
                                        grad_b[i] = output_grad_clone[0] * input_a_data_clone[i];
                                    }
                                }
                                
                                if let Some(input_b_mut) = self.tensors.get_mut(&input_b_id) {
                                    input_b_mut.accumulate_grad(&grad_b);
                                    println!("Debug: Matmul - Updated grad for input_b tensor {}: {:?}", input_b_id, input_b_mut.get_grad());
                                }
                            }
                        }
                    }
                },
                "add" => {
                    // For addition: if z = x + y, then dz/dx = 1, dz/dy = 1
                    if op.inputs.len() >= 2 {
                        let input_a_id = op.inputs[0];
                        let input_b_id = op.inputs[1];
                        
                        // Clone the gradient to avoid borrow checker issues
                        let output_grad_clone = if let Some(output_tensor) = self.tensors.get(&op.output) {
                            if let Some(output_grad) = output_tensor.get_grad() {
                                output_grad.clone()
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        };
                        
                        // Propagate gradient to first input
                        if let Some(input_a) = self.tensors.get(&input_a_id) {
                            if input_a.requires_grad() {
                                if let Some(input_a_mut) = self.tensors.get_mut(&input_a_id) {
                                    input_a_mut.accumulate_grad(&output_grad_clone);
                                    println!("Debug: Add - Updated grad for input_a tensor {}: {:?}", input_a_id, input_a_mut.get_grad());
                                }
                            }
                        }
                        
                        // Propagate gradient to second input
                        if let Some(input_b) = self.tensors.get(&input_b_id) {
                            if input_b.requires_grad() {
                                if let Some(input_b_mut) = self.tensors.get_mut(&input_b_id) {
                                    input_b_mut.accumulate_grad(&output_grad_clone);
                                    println!("Debug: Add - Updated grad for input_b tensor {}: {:?}", input_b_id, input_b_mut.get_grad());
                                }
                            }
                        }
                    }
                },
                "relu" => {
                    // For ReLU: if z = relu(x), then dz/dx = 1 if x > 0, 0 otherwise
                    if !op.inputs.is_empty() {
                        let input_id = op.inputs[0];
                        
                        // Clone the needed data first to avoid borrow checker issues
                        let (input_data_clone, output_grad_clone) = {
                            if let (Some(input), Some(output_tensor)) = (
                                self.tensors.get(&input_id),
                                self.tensors.get(&op.output)
                            ) {
                                if input.requires_grad() {
                                    if let Some(output_grad) = output_tensor.get_grad() {
                                        (input.get_data().clone(), output_grad.clone())
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        // Now use the cloned data to compute gradients
                        let mut relu_grad = vec![0.0; input_data_clone.len()];
                        
                        // Calculate relu gradient
                        for i in 0..input_data_clone.len() {
                            relu_grad[i] = if input_data_clone[i] > 0.0 { 
                                output_grad_clone[i] 
                            } else { 
                                0.0 
                            };
                        }
                        
                        // Finally update the gradient in a separate mutable borrow
                        if let Some(input_mut) = self.tensors.get_mut(&input_id) {
                            input_mut.accumulate_grad(&relu_grad);
                            println!("Debug: ReLU - Updated grad for input tensor {}: {:?}", input_id, input_mut.get_grad());
                        }
                    }
                },
                "sigmoid" => {
                    // For sigmoid: if z = sigmoid(x), then dz/dx = sigmoid(x) * (1 - sigmoid(x))
                    if !op.inputs.is_empty() {
                        let input_id = op.inputs[0];
                        
                        // Clone the needed data first to avoid borrow checker issues
                        let (output_data_clone, output_grad_clone) = {
                            if let (Some(input), Some(output_tensor)) = (
                                self.tensors.get(&input_id),
                                self.tensors.get(&op.output)
                            ) {
                                if input.requires_grad() {
                                    if let Some(output_grad) = output_tensor.get_grad() {
                                        (output_tensor.get_data().clone(), output_grad.clone())
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        // Now compute gradients using cloned data
                        let mut sigmoid_grad = vec![0.0; output_data_clone.len()];
                        
                        // Calculate sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) * upstream_grad
                        for i in 0..output_data_clone.len() {
                            let sigmoid_val = output_data_clone[i];
                            sigmoid_grad[i] = sigmoid_val * (1.0 - sigmoid_val) * output_grad_clone[i];
                        }
                        
                        // Finally, update the gradients in a separate mutable borrow
                        if let Some(input_mut) = self.tensors.get_mut(&input_id) {
                            input_mut.accumulate_grad(&sigmoid_grad);
                            println!("Debug: Sigmoid - Updated grad for input tensor {}: {:?}", input_id, input_mut.get_grad());
                        }
                    }
                },
                "binary_cross_entropy" => {
                    // For binary cross entropy: if L = -(y * log(p) + (1-y) * log(1-p)), then dL/dp = -y/p + (1-y)/(1-p)
                    if op.inputs.len() >= 2 {
                        let pred_id = op.inputs[0]; // Predicted probabilities
                        let target_id = op.inputs[1]; // Target values (0 or 1)
                        
                        // Clone the needed data first to avoid borrow checker issues
                        let (pred_data_clone, target_data_clone, output_grad_clone) = {
                            if let (Some(pred), Some(target), Some(output_tensor)) = (
                                self.tensors.get(&pred_id),
                                self.tensors.get(&target_id),
                                self.tensors.get(&op.output)
                            ) {
                                if pred.requires_grad() {
                                    if let Some(output_grad) = output_tensor.get_grad() {
                                        (
                                            pred.get_data().clone(), 
                                            target.get_data().clone(), 
                                            output_grad.clone()
                                        )
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        // Now compute the gradients using cloned data
                        let mut bce_grad = vec![0.0; pred_data_clone.len()];
                        
                        // Calculate BCE gradient
                        for i in 0..pred_data_clone.len().min(target_data_clone.len()) {
                            let p = pred_data_clone[i].clamp(1e-7, 1.0 - 1e-7); // Clip to avoid division by zero
                            let y = target_data_clone[i];
                            
                            // Gradient of BCE: -y/p + (1-y)/(1-p)
                            bce_grad[i] = (-y / p + (1.0 - y) / (1.0 - p)) * output_grad_clone[0];
                        }
                        
                        // Finally, update the gradients in a separate mutable borrow
                        if let Some(pred_mut) = self.tensors.get_mut(&pred_id) {
                            pred_mut.accumulate_grad(&bce_grad);
                            println!("Debug: BCE - Updated grad for pred tensor {}: {:?}", pred_id, pred_mut.get_grad());
                        }
                    }
                },
                "concat" => {
                    // For concat: Distribute output gradient to all input tensors
                    if !op.inputs.is_empty() {
                        // Clone the output gradient
                        let output_grad_clone = if let Some(output_tensor) = self.tensors.get(&op.output) {
                            if let Some(output_grad) = output_tensor.get_grad() {
                                output_grad.clone()
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        };
                        
                        // Process each input tensor - first collect input sizes
                        let mut input_sizes = Vec::new();
                        for input_id in &op.inputs {
                            if let Some(input_tensor) = self.tensors.get(input_id) {
                                if input_tensor.requires_grad() {
                                    input_sizes.push((input_id, input_tensor.get_data().len()));
                                } else {
                                    input_sizes.push((input_id, 0)); // Mark as not requiring gradients
                                }
                            }
                        }
                        
                        // Now process each input with collected sizes
                        let mut offset = 0;
                        for (input_id, input_size) in input_sizes {
                            if input_size > 0 { // Only process inputs that require gradients
                                // Extract the relevant part of the output gradient for this input
                                let mut input_grad = vec![0.0; input_size];
                                for i in 0..input_size.min(output_grad_clone.len() - offset) {
                                    if offset + i < output_grad_clone.len() {
                                        input_grad[i] = output_grad_clone[offset + i];
                                    }
                                }
                                
                                // Accumulate gradient to the input tensor
                                if let Some(input_mut) = self.tensors.get_mut(input_id) {
                                    input_mut.accumulate_grad(&input_grad);
                                    println!("Debug: Concat - Updated grad for input tensor {}: {:?}", input_id, input_mut.get_grad());
                                }
                            }
                            
                            // Always advance offset by input size
                            offset += input_size;
                        }
                    }
                },
                _ => {
                    println!("Debug: Warning - Unknown operation type: {}", op.op_type);
                }
            }

            // Find operations that produced input tensors and add them to the queue if not visited
            for input_id in &op.inputs {
                for (next_op_id, next_op) in self.operations.iter().enumerate() {
                    if next_op.output == *input_id && !visited_ops.contains(&next_op_id) {
                        op_queue.push_back(next_op_id);
                        visited_ops.insert(next_op_id);
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
    gradient_descent(&mut var, 0.01);
    var.value
}