use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::models::layers::Layer;
use crate::ai::engine::tensor::Tensor;

#[derive(Debug)]
pub struct SGD {
    momentum: f64,
    velocities: Vec<Vec<Vec<f64>>>, // Velocity for each layer's weights
    bias_velocities: Vec<Vec<f64>>, // Velocity for each layer's biases
}

impl SGD {
    pub fn new(momentum: f64, layers: &[Layer]) -> Self {
        let velocities: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .map(|layer| {
                layer.weights.iter().map(|neuron_weights| {
                    vec![0.0; neuron_weights.get_data().len()]
                }).collect()
            })
            .collect();

        let bias_velocities: Vec<Vec<f64>> = layers
            .iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect();

        SGD {
            momentum,
            velocities,
            bias_velocities,
        }
    }
}

pub trait Optimizer {
    fn step(&mut self, model: &mut NeuralNetwork, learning_rate: f64);
}

impl Optimizer for SGD {
    fn step(&mut self, model: &mut NeuralNetwork, learning_rate: f64) {
        for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
            for (neuron_idx, neuron_weights) in layer.weights.iter_mut().enumerate() {
                if let Some(grad_weights) = neuron_weights.get_grad() {
                    let mut weights_data = neuron_weights.get_data().to_vec();
                    println!("Debug: Layer {}, Neuron {}, Before update - Weights: {:?}", layer_idx, neuron_idx, weights_data);
                    println!("Debug: Gradients: {:?}", grad_weights);
                    for (weight_idx, weight) in weights_data.iter_mut().enumerate() {
                        let grad = grad_weights[weight_idx];
                        self.velocities[layer_idx][neuron_idx][weight_idx] = 
                            self.momentum * self.velocities[layer_idx][neuron_idx][weight_idx] 
                            - learning_rate * grad;
                        *weight += self.velocities[layer_idx][neuron_idx][weight_idx];
                    }
                    println!("Debug: Layer {}, Neuron {}, After update - Weights: {:?}", layer_idx, neuron_idx, weights_data);
                    let len = weights_data.len();
                    *neuron_weights = Tensor::with_grad(weights_data, vec![len]);
                }
            }

            for (bias_idx, bias) in layer.biases.iter_mut().enumerate() {
                if let Some(grad_biases) = bias.get_grad() {
                    let mut bias_data = bias.get_data().to_vec();
                    println!("Debug: Layer {}, Bias {}, Before update - Bias: {:?}", layer_idx, bias_idx, bias_data);
                    println!("Debug: Bias Gradients: {:?}", grad_biases);
                    let grad = grad_biases[0];
                    self.bias_velocities[layer_idx][bias_idx] = 
                        self.momentum * self.bias_velocities[layer_idx][bias_idx] 
                        - learning_rate * grad;
                    bias_data[0] += self.bias_velocities[layer_idx][bias_idx];
                    println!("Debug: Layer {}, Bias {}, After update - Bias: {:?}", layer_idx, bias_idx, bias_data);
                    let len = bias_data.len();
                    *bias = Tensor::with_grad(bias_data, vec![len]);
                }
            }
        }
    }
}