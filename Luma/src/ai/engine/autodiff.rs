use crate::ai::engine::tensor::Tensor;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct Operation {
    op_type: String,
    inputs: Vec<usize>,
    output: usize,
    metadata: HashMap<String, String>,
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
                    println!("Debug: Registered input tensor for op {} with ID {}", op_type, registered.id);
                    registered.id
                } else {
                    println!("Debug: Input tensor for op {} already registered with ID {}", op_type, t.id);
                    t.id
                }
            })
            .collect();

        let output_id = if output.id == 0 {
            let registered = self.register_tensor(output);
            println!("Debug: Registered output tensor for op {} with ID {}", op_type, registered.id);
            registered.id
        } else {
            println!("Debug: Output tensor for op {} already registered with ID {}", op_type, output.id);
            output.id
        };

        let op_id = self.operations.len();
        self.operations.push(Operation {
            op_type: op_type.to_string(),
            inputs: input_ids.clone(),
            output: output_id,
            metadata: HashMap::new(),
        });
        println!("Debug: Added operation {} (type: {}), inputs: {:?}, output: {}", op_id, op_type, input_ids, output_id);

        op_id
    }

    pub fn set_op_metadata(&mut self, op_id: usize, key: &str, value: String) {
        if op_id < self.operations.len() {
            self.operations[op_id].metadata.insert(key.to_string(), value);
        }
    }

    pub fn backward(&mut self, output_id: usize) {
        // Initialize gradient of the output tensor
        if let Some(tensor) = self.tensors.get_mut(&output_id) {
            tensor.set_grad(&vec![1.0]); // Start with gradient of 1.0 for the output tensor
            println!("Debug: Initial grad for output tensor {}: {:?}", output_id, tensor.get_grad());
        } else {
            println!("Debug: Error - Output tensor {} not found", output_id);
            return;
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
                "matmul" => { /* Implementation for matmul would go here */ },
                "add" => { /* Implementation for add would go here */ },
                "relu" => { /* Implementation for relu would go here */ },
                "sigmoid" => { /* Implementation for sigmoid would go here */ },
                "binary_cross_entropy" => { /* Implementation for binary_cross_entropy would go here */ },
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