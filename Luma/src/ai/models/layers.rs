use rand::Rng;
use crate::ai::engine::autodiff::ComputationGraph;
use crate::ai::engine::tensor::Tensor;

#[derive(Debug)]
pub struct Layer {
    pub id: String,
    pub neurons: usize,
    pub weights: Vec<Tensor>,
    pub biases: Vec<Tensor>,
    pub last_input: Vec<f64>,
    pub last_output: Vec<f64>,
    pub is_output_layer: bool,
}

impl Layer {
    pub fn new(id: String, input_size: usize, neurons: usize, is_output_layer: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + neurons) as f64).sqrt();
        let mut weights = Vec::with_capacity(neurons);
        let mut biases = Vec::with_capacity(neurons);

        for _ in 0..neurons {
            let mut neuron_weights = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                neuron_weights.push(rng.gen_range(-scale..scale));
            }
            weights.push(Tensor::with_grad(neuron_weights, vec![input_size]));
            biases.push(Tensor::with_grad(vec![0.0], vec![1]));
        }

        Layer {
            id,
            neurons,
            weights,
            biases,
            last_input: vec![0.0; input_size],
            last_output: vec![0.0; neurons],
            is_output_layer,
        }
    }

    pub fn forward(&mut self, input: Tensor, graph: &mut ComputationGraph) -> Tensor {
        let input_data = input.get_data().to_vec();
        self.last_input = input_data.clone();
        let mut output = Vec::with_capacity(self.neurons);

        let input_tensor = if input.id == 0 {
            let registered = graph.register_tensor(input);
            println!("Debug: Registered input tensor with ID {}", registered.id);
            registered
        } else {
            println!("Debug: Input tensor already registered with ID {}", input.id);
            input
        };

        for i in 0..self.neurons {
            let weight_tensor = &self.weights[i];
            let bias_tensor = &self.biases[i];

            // Ensure weights and biases are registered
            assert!(weight_tensor.id != 0, "Weight tensor not registered for neuron {}", i);
            assert!(bias_tensor.id != 0, "Bias tensor not registered for neuron {}", i);
            println!("Debug: Using weight tensor ID {} and bias tensor ID {} for neuron {}", 
                     weight_tensor.id, bias_tensor.id, i);

            // Compute matmul: sum = input * weights
            let mut sum = 0.0;
            for j in 0..input_data.len() {
                sum += input_data[j] * weight_tensor.get_data()[j];
            }
            let sum_tensor = Tensor::with_grad(vec![sum], vec![1]);
            let sum_tensor = graph.register_tensor(sum_tensor);
            graph.add_operation("matmul", vec![input_tensor.clone(), weight_tensor.clone()], sum_tensor.clone());
            println!("Debug: Matmul operation for neuron {}, sum tensor ID {}", i, sum_tensor.id);

            // Add bias: sum += bias
            sum += bias_tensor.get_data()[0];
            let sum_with_bias = Tensor::with_grad(vec![sum], vec![1]);
            let sum_with_bias = graph.register_tensor(sum_with_bias);
            graph.add_operation("add", vec![sum_tensor, bias_tensor.clone()], sum_with_bias.clone());
            println!("Debug: Add operation for neuron {}, sum_with_bias tensor ID {}", i, sum_with_bias.id);

            // Apply activation
            let activated_value = if self.is_output_layer {
                sigmoid(sum)
            } else {
                sum.max(0.0)
            };
            self.last_output[i] = activated_value;

            let activated_tensor = Tensor::with_grad(vec![activated_value], vec![1]);
            let activated_tensor = graph.register_tensor(activated_tensor);
            let op_type = if self.is_output_layer { "sigmoid" } else { "relu" };
            graph.add_operation(op_type, vec![sum_with_bias.clone()], activated_tensor.clone());
            println!("Debug: {} operation for neuron {}, activated tensor ID {}", op_type, i, activated_tensor.id);

            output.push(activated_value);
        }

        let output_tensor = Tensor::with_grad(output, vec![self.neurons]);
        let output_tensor = graph.register_tensor(output_tensor);
        println!("Debug: Final output tensor ID {} for layer {}", output_tensor.id, self.id);
        
        // Link layer output to individual neurons for better graph connectivity
        // Create a concat operation that pulls together the individual neuron activations
        let mut input_tensors = Vec::with_capacity(self.neurons);
        for i in 0..self.neurons {
            let neuron_id = self.neurons - i - 1; // Get most recent operations first
            let op = graph.operations.iter().rev()
                .find(|op| op.op_type == (if self.is_output_layer { "sigmoid" } else { "relu" }) 
                     && op.inputs.len() > 0);
            
            if let Some(op) = op {
                let activated_tensor_id = op.output;
                if let Some(tensor) = graph.tensors.get(&activated_tensor_id) {
                    input_tensors.push(tensor.clone());
                }
            }
        }
        
        // Only add the concat operation if we found all neuron activations
        if !input_tensors.is_empty() {
            graph.add_operation("concat", input_tensors, output_tensor.clone());
            println!("Debug: Added concat operation to link layer output to neuron activations");
        }
        
        output_tensor
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}