use std::cell::RefCell;
use std::fmt::Display;

/// Trait for handling callbacks during model training
pub trait Callback {
    fn on_epoch_begin(&self, _epoch: usize) {}
    fn on_epoch_end(&self, epoch: usize, loss: f64);
    fn on_training_begin(&self) {}
    fn on_training_end(&self, _final_loss: f64) {}
    fn on_batch_begin(&self, _batch: usize) {}
    fn on_batch_end(&self, _batch: usize, _loss: f64) {}
}

/// A simple logging callback that prints training progress
pub struct LoggingCallback {
    verbose: bool,
}

impl LoggingCallback {
    pub fn new(verbose: bool) -> Self {
        LoggingCallback { verbose }
    }
}

impl Default for LoggingCallback {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Callback for LoggingCallback {
    fn on_training_begin(&self) {
        if self.verbose {
            println!("Training started");
        }
    }

    fn on_epoch_begin(&self, epoch: usize) {
        if self.verbose {
            println!("Epoch {} started", epoch + 1);
        }
    }

    fn on_epoch_end(&self, epoch: usize, loss: f64) {
        println!("Logging - Epoch {}: Loss = {:.6}", epoch + 1, loss);
    }

    fn on_training_end(&self, final_loss: f64) {
        println!("Training completed. Final loss: {:.6}", final_loss);
    }
}

/// Early stopping callback to prevent overfitting
pub struct EarlyStoppingCallback {
    patience: usize,
    min_delta: f64,
    counter: RefCell<usize>,
    best_loss: RefCell<f64>,
    stopped: RefCell<bool>,
}

impl EarlyStoppingCallback {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        EarlyStoppingCallback {
            patience,
            min_delta,
            counter: RefCell::new(0),
            best_loss: RefCell::new(f64::INFINITY),
            stopped: RefCell::new(false),
        }
    }

    pub fn should_stop(&self) -> bool {
        *self.stopped.borrow()
    }
}

impl Callback for EarlyStoppingCallback {
    fn on_epoch_end(&self, epoch: usize, loss: f64) {
        if loss < *self.best_loss.borrow() - self.min_delta {
            // Loss improved
            *self.best_loss.borrow_mut() = loss;
            *self.counter.borrow_mut() = 0;
        } else {
            // Loss didn't improve enough
            *self.counter.borrow_mut() += 1;

            if *self.counter.borrow() >= self.patience {
                *self.stopped.borrow_mut() = true;
                println!("Early stopping triggered after {} epochs", epoch + 1);
            }
        }
    }
}

/// ModelCheckpoint callback to save the best model
pub struct ModelCheckpointCallback<T: Display> {
    model_id: T,
    best_loss: RefCell<f64>,
    save_best_only: bool,
}

impl<T: Display> ModelCheckpointCallback<T> {
    pub fn new(model_id: T, save_best_only: bool) -> Self {
        ModelCheckpointCallback {
            model_id,
            best_loss: RefCell::new(f64::INFINITY),
            save_best_only,
        }
    }
}

impl<T: Display> Callback for ModelCheckpointCallback<T> {
    fn on_epoch_end(&self, epoch: usize, loss: f64) {
        if !self.save_best_only || loss < *self.best_loss.borrow() {
            *self.best_loss.borrow_mut() = loss;
            println!("Saving model {} after epoch {} (loss: {:.6})", self.model_id, epoch + 1, loss);
            // Here you would typically call model saving logic
            // e.g., save_model(self.model_id);
        }
    }
}

/// A collection of callbacks to be executed together
pub struct CallbackList {
    callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackList {
    pub fn new() -> Self {
        CallbackList {
            callbacks: Vec::new(),
        }
    }

    pub fn add<C: Callback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }
}

impl Callback for CallbackList {
    fn on_training_begin(&self) {
        for callback in &self.callbacks {
            callback.on_training_begin();
        }
    }

    fn on_epoch_begin(&self, epoch: usize) {
        for callback in &self.callbacks {
            callback.on_epoch_begin(epoch);
        }
    }

    fn on_epoch_end(&self, epoch: usize, loss: f64) {
        for callback in &self.callbacks {
            callback.on_epoch_end(epoch, loss);
        }
    }

    fn on_training_end(&self, final_loss: f64) {
        for callback in &self.callbacks {
            callback.on_training_end(final_loss);
        }
    }

    fn on_batch_begin(&self, batch: usize) {
        for callback in &self.callbacks {
            callback.on_batch_begin(batch);
        }
    }

    fn on_batch_end(&self, batch: usize, loss: f64) {
        for callback in &self.callbacks {
            callback.on_batch_end(batch, loss);
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_set_callback(_max_epochs: i32) -> i32 {
    let _callback = LoggingCallback::default();
    // Setup logic would go here
    0 // Success
}