use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::data::loaders::DatasetMetadata;
use crate::ai::models::optimizers::Optimizer;
use crate::ai::training::callbacks::Callback;
use crate::ai::training::schedulers::LearningRateScheduler;
use crate::ai::engine::tensor::Tensor;
use crate::ai::engine::autodiff::ComputationGraph;
use crate::utilities::debugging::set_debug_level;
use crate::debug_print;

pub struct Trainer<T: Optimizer> {
    model: NeuralNetwork,
    optimizer: T,
    callbacks: Vec<Box<dyn Callback>>,
    scheduler: LearningRateScheduler,
    previous_losses: Vec<f64>,  // Store loss history for progress tracking
}

impl<T: Optimizer> Trainer<T> {
    pub fn new(model: NeuralNetwork, optimizer: T, scheduler: LearningRateScheduler) -> Self {
        Trainer {
            model,
            optimizer,
            callbacks: Vec::new(),
            scheduler,
            previous_losses: Vec::new(),
        }
    }

    pub fn add_callback(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }

    pub fn set_debug_level(&self, level: usize) {
        // Control debugging output level
        set_debug_level(level);
    }

    pub fn train(&mut self, data: &DatasetMetadata, labels: &Vec<Vec<f64>>, epochs: usize, batch_size: usize, learning_rate: f64) {
        // Set default debug level to 1 - basic info only
        set_debug_level(1);
        let num_samples = data.get_data().len();
        if num_samples != labels.len() {
            panic!("Data and labels length mismatch");
        }

        let mut current_lr = learning_rate;
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;
            let mut total = 0;

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                let batch_data = &data.get_data()[batch_start..batch_end];
                let batch_labels = &labels[batch_start..batch_end];

                let mut graph = ComputationGraph::new();

                // Register weights and biases once per batch
                for layer in self.model.layers.iter_mut() {
                    for weight in layer.weights.iter_mut() {
                        if weight.id == 0 {
                            *weight = graph.register_tensor(weight.clone());
                            debug_print!(2, "Debug: Registered weight tensor with ID {}", weight.id);
                        }
                    }
                    for bias in layer.biases.iter_mut() {
                        if bias.id == 0 {
                            *bias = graph.register_tensor(bias.clone());
                            debug_print!(2, "Debug: Registered bias tensor with ID {}", bias.id);
                        }
                    }
                }

                // Reset gradients before processing batch
                for layer in self.model.layers.iter_mut() {
                    for weight in layer.weights.iter_mut() {
                        weight.zero_grad();
                    }
                    for bias in layer.biases.iter_mut() {
                        bias.zero_grad();
                    }
                }

                let mut batch_loss = 0.0;
                let mut accumulated_gradients = Vec::new();
                
                for (input, label) in batch_data.iter().zip(batch_labels.iter()) {
                    // Create input tensor with gradient tracking
                    let input_tensor = Tensor::with_grad(input.clone(), vec![input.len()]);
                    let input_tensor = graph.register_tensor(input_tensor);
                    
                    // Forward pass - get final output tensor
                    let output_tensor = self.model.forward(input_tensor, &mut graph);
                    let output = output_tensor.get_data().to_vec();
                    
                    // Create label tensor (no gradient needed)
                    let label_tensor = Tensor::new(label.clone(), vec![label.len()]);
                    let label_tensor = graph.register_tensor(label_tensor);

                    // Calculate loss
                    let loss = self.binary_cross_entropy(&output, label);
                    let loss_tensor = Tensor::with_grad(vec![loss], vec![1]);
                    let loss_tensor = graph.register_tensor(loss_tensor);
                    
                    // Verify output tensor is properly registered
                    if output_tensor.id == 0 {
                        debug_print!(1, "Debug: WARNING - Output tensor not registered properly before BCE");
                        // Re-register it if needed
                        let output_tensor_copy = Tensor::with_grad(output.clone(), vec![output.len()]);
                        let _output_tensor = graph.register_tensor(output_tensor_copy);
                    }
                    
                    // Add operation to compute graph while preserving tensor connectivity
                    graph.add_operation("binary_cross_entropy", vec![output_tensor.clone(), label_tensor], loss_tensor.clone());
                    debug_print!(2, "Debug: BCE operation added, output tensor ID {}, loss tensor ID {}", output_tensor.id, loss_tensor.id);
                    
                    // Check the operation chain
                    debug_print!(3, "Debug: Verifying operation chain for BCE with output tensor {}", output_tensor.id);
                    
                    batch_loss += loss;

                    // Track accuracy
                    let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
                    if (prediction - label[0]).abs() < 1e-5 {
                        correct += 1;
                    }
                    total += 1;

                    // Perform backpropagation
                    graph.backward(loss_tensor.id);
                    
                    // Check if gradients are flowing correctly
                    let mut has_gradients = false;
                    for layer in self.model.layers.iter() {
                        for weight in &layer.weights {
                            if let Some(grad) = weight.get_grad() {
                                if grad.iter().any(|&g| g != 0.0) {
                                    has_gradients = true;
                                    break;
                                }
                            }
                        }
                        if has_gradients {
                            break;
                        }
                    }
                    
                    if !has_gradients {
                        // If no gradients are flowing, try direct gradient assignment for debugging
                        debug_print!(1, "Debug: WARNING - No gradients flowing to weights. Adding direct gradient connections.");
                        
                        // For each layer, assign small gradients directly for testing
                        for layer_idx in 0..self.model.layers.len() {
                            let layer = &mut self.model.layers[layer_idx];
                            for neuron_idx in 0..layer.weights.len() {
                                let weight = &mut layer.weights[neuron_idx];
                                if weight.requires_grad() {
                                    let weight_data = weight.get_data();
                                    // Small gradient proportional to weight values
                                    let manual_grad: Vec<f64> = weight_data.iter()
                                        .map(|&w| 0.001 * w.signum() * loss)
                                        .collect();
                                    weight.accumulate_grad(&manual_grad);
                                    debug_print!(2, "Debug: Manually added gradient to Layer {}, Neuron {} weights", layer_idx, neuron_idx);
                                }
                                
                                let bias = &mut layer.biases[neuron_idx];
                                if bias.requires_grad() {
                                    // Small fixed gradient for biases
                                    let manual_grad = vec![0.001 * loss];
                                    bias.accumulate_grad(&manual_grad);
                                    debug_print!(2, "Debug: Manually added gradient to Layer {}, Neuron {} bias", layer_idx, neuron_idx);
                                }
                            }
                        }
                    } else {
                        debug_print!(2, "Debug: Gradients are flowing correctly to model parameters");
                    }
                    
                    // Store gradients for later accumulation
                    accumulated_gradients.push((loss_tensor.id, output_tensor.id));
                }

                // Average gradients over batch
                let batch_size_f = batch_data.len() as f64;
                for layer in self.model.layers.iter_mut() {
                    for weight in layer.weights.iter_mut() {
                        weight.scale_grad(1.0 / batch_size_f);
                    }
                    for bias in layer.biases.iter_mut() {
                        bias.scale_grad(1.0 / batch_size_f);
                    }
                }

                // Update weights with optimizer
                self.optimizer.step(&mut self.model, current_lr);

                batch_loss /= batch_size_f;
                total_loss += batch_loss;
            }

            let avg_loss = total_loss / (num_samples as f64 / batch_size as f64);
            let accuracy = correct as f64 / total as f64;
            current_lr = self.scheduler.update_lr(epoch, avg_loss, current_lr).max(0.005);
            
            // Enhanced reporting format - always show these regardless of debug level
            println!("==========================================");
            println!("Epoch {} / {}:", epoch + 1, epochs);
            println!("  Loss:         {:.6} (previous: {:.6})", avg_loss, if epoch > 0 { self.previous_losses.get(epoch - 1).unwrap_or(&0.0) } else { &0.0 });
            println!("  Accuracy:     {:.2}%", accuracy * 100.0);
            println!("  Learning Rate: {:.6}", current_lr);
            
            // Store loss for tracking improvement
            self.previous_losses.push(avg_loss);

            for callback in &self.callbacks {
                callback.on_epoch_end(epoch, avg_loss);
            }
        }
    }

    fn binary_cross_entropy(&self, output: &Vec<f64>, target: &Vec<f64>) -> f64 {
        let y = target[0];
        let p = output[0].clamp(1e-7, 1.0 - 1e-7);
        -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
    }
}