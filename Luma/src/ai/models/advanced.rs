use crate::ai::models::layers::Layer;
use crate::ai::engine::autodiff::ComputationGraph;
use crate::ai::engine::tensor::Tensor;

#[derive(Debug)]
pub struct NeuralNetwork {
    pub id: String,
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(id: &str, layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let is_output_layer = i == layer_sizes.len() - 2;
            let layer = Layer::new(
                format!("layer_{}", i),
                layer_sizes[i],
                layer_sizes[i + 1],
                is_output_layer,
            );
            layers.push(layer);
        }
        NeuralNetwork {
            id: id.to_string(),
            layers,
        }
    }

    pub fn forward(&mut self, input: Tensor, graph: &mut ComputationGraph) -> Tensor {
        let mut output = input;
        // Pass through each layer sequentially
        for layer in &mut self.layers {
            // Layer's forward method should maintain tensor connectivity
            output = layer.forward(output, graph);
        }
        // Return the final output tensor without modifying it
        // Critical to preserve the computational graph for backpropagation
        output
    }
}