use luma::ai_data::loaders;
use luma::ai_data::preprocessors::{self, PreprocessingMethod};
use luma::ai_data::augmentations::{self, AugmentationMethod};
use luma::ai_data::multi_modal;

fn main() {
    println!("Luma Data Loading Example");
    println!("=========================\n");

    // Basic CSV loading
    println!("1. Loading CSV data:");
    match loaders::load_dataset("samples/iris.csv", "iris", false, None) {
        Ok(_) => println!("  ✅ Loaded iris dataset successfully"),
        Err(e) => println!("  ❌ Failed to load iris dataset: {}", e),
    }

    // Print the loaded dataset
    println!("\n2. Viewing dataset:");
    loaders::print_dataset("iris");

    // Lazy loading example
    println!("\n3. Lazy loading example:");
    match loaders::load_dataset("samples/iris.csv", "iris_lazy", true, None) {
        Ok(_) => println!("  ✅ Configured iris_lazy dataset for lazy loading"),
        Err(e) => println!("  ❌ Failed to configure lazy loading: {}", e),
    }

    // Access the lazy-loaded dataset (will load it on first access)
    println!("\n4. Accessing lazy-loaded dataset:");
    if let Some(dataset) = loaders::get_dataset("iris_lazy") {
        println!("  ✅ Lazy dataset loaded on demand");
        println!("  Number of rows: {}", dataset.get_data().len());
    } else {
        println!("  ❌ Failed to access lazy dataset");
    }

    // Data preprocessing example
    println!("\n5. Data preprocessing example:");
    match preprocessors::preprocess_dataset("iris", PreprocessingMethod::Normalize, Some("iris_normalized")) {
        Ok(name) => println!("  ✅ Created normalized dataset '{}'", name),
        Err(e) => println!("  ❌ Failed to normalize dataset: {}", e),
    }
    
    // Data augmentation example
    println!("\n6. Data augmentation example:");
    match augmentations::augment_dataset("iris", "iris_augmented", AugmentationMethod::Noise(0.1)) {
        Ok(_) => println!("  ✅ Created augmented dataset with added noise"),
        Err(e) => println!("  ❌ Failed to augment dataset: {}", e),
    }
    
    // Dataset splitting example
    println!("\n7. Dataset splitting example:");
    match loaders::split_dataset("iris", 0.3) {
        Ok((train, test)) => println!("  ✅ Split iris dataset into '{}' and '{}'", train, test),
        Err(e) => println!("  ❌ Failed to split dataset: {}", e),
    }
    
    // Multi-modal data example (if a sample image is available)
    println!("\n8. Multi-modal data example:");
    if let Ok(_) = multi_modal::load_dataset("samples/config.json", "config_data") {
        println!("  ✅ Loaded JSON configuration data");
        if let Some(data) = multi_modal::get_dataset("config_data") {
            println!("  Data type: {}", data.get_type());
        }
    } else {
        println!("  ❌ Failed to load multi-modal data");
    }
    
    println!("\n=========================");
    println!("Example completed!");
}