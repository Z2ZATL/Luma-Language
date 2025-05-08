use std::io::{self, Write};
use crate::ai::data::{loaders, augmentations};
use crate::ai::training::trainers::Trainer;
use crate::ai::models::advanced::NeuralNetwork;
use crate::ai::models::optimizers::SGD;
use crate::ai::training::schedulers::LearningRateScheduler;
use crate::ai::training::callbacks::{LoggingCallback, CallbackList};

pub fn start_repl() {
    println!("Luma REPL (type 'exit' to quit)");
    loop {
        print!("Luma> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" {
            println!("Exiting Luma REPL...");
            break;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(0).copied() {
            Some("load") => {
                if parts.len() >= 4 && parts[1] == "dataset" && parts[3] == "as" {
                    let path = parts[2].trim_matches('"');
                    let name = parts[4..].join(" ").trim().to_string();
                    match loaders::load_dataset(path, &name) {
                        Ok(()) => println!("Dataset {} loaded successfully", name),
                        Err(e) => println!("Error loading dataset: {}", e),
                    }
                } else {
                    println!("Usage: load dataset \"path\" as name");
                }
            }
            Some("print_dataset") => {
                if parts.len() == 2 {
                    let name = parts[1].trim_matches('"');
                    loaders::print_dataset(name);
                } else {
                    println!("Usage: print_dataset \"name\"");
                }
            }
            Some("augment") => {
                if parts.len() >= 6 && parts[1] == "dataset" && parts[3] == "as" && parts[5].starts_with("lazy=") {
                    let path = parts[2].trim_matches('"');
                    let name = parts[4];
                    let lazy_str = parts[5].split('=').nth(1).unwrap_or("false");
                    let lazy = lazy_str == "True" || lazy_str == "true";
                    match augmentations::augment_dataset(path, name, lazy) {
                        Ok(()) => println!("Dataset {} augmented successfully", name),
                        Err(e) => println!("Error augmenting dataset: {}", e),
                    }
                } else {
                    println!("Usage: augment dataset \"path\" as name lazy=True|False");
                }
            }
            Some("print_augmentation") => {
                if parts.len() == 2 {
                    let name = parts[1].trim_matches('"');
                    augmentations::print_augmentation(name);
                } else {
                    println!("Usage: print_augmentation \"name\"");
                }
            }
            Some("train") => {
                let command = parts[1..].join(" ");
                let params: Vec<&str> = command.split_whitespace().collect();

                let mut epochs = 0;
                let mut batch_size = 0;
                let mut learning_rate = 0.0;
                let mut valid = true;

                for param in params {
                    let kv: Vec<&str> = param.split('=').collect();
                    if kv.len() != 2 {
                        valid = false;
                        break;
                    }
                    match kv[0] {
                        "epochs" => epochs = kv[1].parse().unwrap_or(0),
                        "batch_size" => batch_size = kv[1].parse().unwrap_or(0),
                        "learning_rate" => learning_rate = kv[1].parse().unwrap_or(0.0),
                        _ => valid = false,
                    }
                }

                if valid && epochs > 0 && batch_size > 0 && learning_rate > 0.0 {
                    if let Some(dataset) = loaders::get_dataset("training_data") {
                        let layer_sizes = vec![2, 8, 4, 1];
                        let model = NeuralNetwork::new("nn_model", layer_sizes);
                        let optimizer = SGD::new(0.9, &model.layers);
                        let scheduler = LearningRateScheduler::new(0.01); // Reduced decay rate

                        let mut trainer = Trainer::new(model, optimizer, scheduler);
                        let mut callbacks = CallbackList::new();
                        callbacks.add(LoggingCallback::new(true));
                        trainer.add_callback(Box::new(callbacks));

                        trainer.train(&dataset, dataset.get_labels(), epochs, batch_size, 0.01); // Initial LR = 0.01
                    } else {
                        println!("Dataset 'training_data' not found");
                    }
                } else {
                    println!("Usage: train epochs=<num> batch_size=<num> learning_rate=<num>");
                }
            }
            _ => println!("Unknown command: {}. Type 'exit' to quit.", input),
        }
        io::stdout().flush().unwrap();
    }
}