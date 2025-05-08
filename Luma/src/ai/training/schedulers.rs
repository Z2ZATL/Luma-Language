/// LearningRateScheduler for adjusting learning rate during training
pub struct LearningRateScheduler {
    strategy: SchedulingStrategy,
}

/// Different learning rate scheduling strategies
pub enum SchedulingStrategy {
    TimeBasedDecay {
        decay: f64,
    },
    StepDecay {
        step_size: usize,
        gamma: f64,
    },
    ExponentialDecay {
        decay: f64,
    },
    CosineAnnealing {
        t_max: usize,
        eta_min: f64,
    },
    ReduceOnPlateau {
        patience: usize,
        factor: f64,
        best_loss: f64,
        wait: usize,
    },
    Constant,
}

impl LearningRateScheduler {
    /// Creates a new learning rate scheduler with time-based decay
    pub fn new(decay: f64) -> Self {
        LearningRateScheduler { 
            strategy: SchedulingStrategy::TimeBasedDecay { decay }
        }
    }

    /// Creates a scheduler with step decay strategy
    pub fn step_decay(step_size: usize, gamma: f64) -> Self {
        LearningRateScheduler {
            strategy: SchedulingStrategy::StepDecay { step_size, gamma }
        }
    }

    /// Creates a scheduler with exponential decay strategy
    pub fn exponential_decay(decay: f64) -> Self {
        LearningRateScheduler {
            strategy: SchedulingStrategy::ExponentialDecay { decay }
        }
    }

    /// Creates a scheduler with cosine annealing strategy
    pub fn cosine_annealing(t_max: usize, eta_min: f64) -> Self {
        LearningRateScheduler {
            strategy: SchedulingStrategy::CosineAnnealing { t_max, eta_min }
        }
    }

    /// Creates a scheduler that reduces learning rate on plateau
    pub fn reduce_on_plateau(patience: usize, factor: f64) -> Self {
        LearningRateScheduler {
            strategy: SchedulingStrategy::ReduceOnPlateau {
                patience,
                factor,
                best_loss: f64::INFINITY,
                wait: 0
            }
        }
    }

    /// Gets the learning rate for the current epoch
    pub fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        match &self.strategy {
            SchedulingStrategy::TimeBasedDecay { decay } => {
                base_lr / (1.0 + decay * epoch as f64)
            },
            SchedulingStrategy::StepDecay { step_size, gamma } => {
                base_lr * gamma.powf((epoch / step_size) as f64)
            },
            SchedulingStrategy::ExponentialDecay { decay } => {
                base_lr * decay.powf(epoch as f64)
            },
            SchedulingStrategy::CosineAnnealing { t_max, eta_min } => {
                let t_max = *t_max as f64;
                let eta_min = *eta_min;
                let epoch = epoch as f64 % t_max;

                eta_min + 0.5 * (base_lr - eta_min) * (1.0 + (std::f64::consts::PI * epoch / t_max).cos())
            },
            SchedulingStrategy::ReduceOnPlateau { .. } => {
                // For reduce on plateau, we need to use update_lr with loss information
                base_lr
            },
            SchedulingStrategy::Constant => base_lr,
        }
    }

    /// Updates learning rate based on current loss (for ReduceOnPlateau)
    pub fn update_lr(&mut self, epoch: usize, loss: f64, current_lr: f64) -> f64 {
        if let SchedulingStrategy::ReduceOnPlateau { patience, factor, best_loss, wait } = &mut self.strategy {
            if loss < *best_loss {
                *best_loss = loss;
                *wait = 0;
                current_lr
            } else {
                *wait += 1;
                if *wait >= *patience {
                    *wait = 0;
                    let new_lr = current_lr * *factor;
                    println!("Reducing learning rate to {} after epoch {}", new_lr, epoch + 1);
                    new_lr
                } else {
                    current_lr
                }
            }
        } else {
            self.get_lr(epoch, current_lr)
        }
    }
}

/// Cyclic learning rate scheduler that varies LR between bounds
pub struct CyclicLRScheduler {
    base_lr: f64,
    max_lr: f64,
    step_size: usize,
    mode: CyclicLRMode,
    current_step: usize,
}

/// Different modes for cyclic learning rate
pub enum CyclicLRMode {
    Triangular,
    Triangular2,
    ExpRange(f64), // With gamma parameter
}

impl CyclicLRScheduler {
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize, mode: CyclicLRMode) -> Self {
        CyclicLRScheduler {
            base_lr,
            max_lr,
            step_size,
            mode,
            current_step: 0,
        }
    }

    pub fn get_lr(&mut self) -> f64 {
        let cycle = (self.current_step / (2 * self.step_size)) as f64;
        let x = (self.current_step % (2 * self.step_size)) as f64;
        let x = if x < self.step_size as f64 {
            x / self.step_size as f64
        } else {
            2.0 - x / self.step_size as f64
        };

        let lr = match &self.mode {
            CyclicLRMode::Triangular => {
                self.base_lr + (self.max_lr - self.base_lr) * x
            },
            CyclicLRMode::Triangular2 => {
                self.base_lr + (self.max_lr - self.base_lr) * x / (2.0_f64.powf(cycle))
            },
            CyclicLRMode::ExpRange(gamma) => {
                self.base_lr + (self.max_lr - self.base_lr) * x * gamma.powf(self.current_step as f64)
            }
        };

        self.current_step += 1;
        lr
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Helper function to adjust learning rate based on epoch
pub fn adjust_learning_rate(epoch: i32) -> f64 {
    0.01 / (1.0 + 0.001 * epoch as f64) // Simple decay
}

#[no_mangle]
pub extern "C" fn luma_adjust_learning_rate(epoch: i32) -> f64 {
    adjust_learning_rate(epoch)
}