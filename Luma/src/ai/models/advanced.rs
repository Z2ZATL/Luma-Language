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
        
        // Ensure the output tensor is registered with the graph
        if output.id == 0 {
            debug_print!(1, "Debug: Registering final output tensor in NeuralNetwork::forward");
            output = graph.register_tensor(output);
        } else {
            debug_print!(3, "Debug: Final output tensor already registered with ID {}", output.id);
        }
        
        // Print debug info to verify final tensor ID
        debug_print!(2, "Debug: NeuralNetwork::forward final output tensor ID: {}", output.id);
        
        // Return the final output tensor without re-registering
        output
    }
}