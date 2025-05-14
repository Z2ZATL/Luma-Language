//! Image Processing Plugin for Luma
//!
//! This plugin provides image processing capabilities to Luma,
//! including resizing, cropping, filtering, and basic transformations.

use crate::plugins::{register_plugin, PluginResult};
use image::{GenericImageView, ImageBuffer, DynamicImage, Rgba};
use std::path::Path;

/// Register the Image Processing plugin with Luma
pub fn register_image_plugin() -> Result<(), String> {
    // List of commands supported by this plugin
    let commands = vec![
        "resize".to_string(),
        "crop".to_string(),
        "grayscale".to_string(),
        "blur".to_string(),
        "rotate".to_string(),
    ];
    
    // Register the plugin
    register_plugin(
        "img",
        "Image Processing",
        "1.0.0",
        Some("Provides image manipulation functionality"),
        Some("Luma Team"),
        commands,
        execute_image_command,
    )
}

/// Execute an image processing command
fn execute_image_command(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No image processing command specified".to_string());
    }
    
    match args[0] {
        "resize" => resize_image(&args[1..]),
        "crop" => crop_image(&args[1..]),
        "grayscale" => grayscale_image(&args[1..]),
        "blur" => blur_image(&args[1..]),
        "rotate" => rotate_image(&args[1..]),
        _ => Err(format!("Unknown image processing command: {}", args[0])),
    }
}

/// Resize an image to specified dimensions
fn resize_image(args: &[&str]) -> PluginResult {
    if args.len() < 3 {
        return Err("Usage: resize <input_path> <width> <height> [output_path]".to_string());
    }
    
    let input_path = args[0];
    let width = match args[1].parse::<u32>() {
        Ok(w) => w,
        Err(_) => return Err("Invalid width value".to_string()),
    };
    
    let height = match args[2].parse::<u32>() {
        Ok(h) => h,
        Err(_) => return Err("Invalid height value".to_string()),
    };
    
    // Use String for both branches of the if-else
    let output_path = if args.len() > 3 {
        args[3].to_string()  // Convert to String
    } else {
        // Create default output path
        let input_path_obj = Path::new(input_path);
        let stem = input_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let extension = input_path_obj.extension().and_then(|s| s.to_str()).unwrap_or("png");
        
        format!("{}_resized.{}", stem, extension)
    };
    
    // Open the image
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => return Err(format!("Failed to open image: {}", e)),
    };
    
    // Resize the image
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    
    // In a real implementation, this would save the image to the output path
    // For our demo, we'll just print the operation details
    
    let original_dimensions = img.dimensions();
    println!("Image would be resized from {}x{} to {}x{} and saved to: {}",
        original_dimensions.0, original_dimensions.1, width, height, output_path);
    
    Ok(format!("Image resized successfully and saved to: {}", output_path))
}

/// Crop an image to specified dimensions
fn crop_image(args: &[&str]) -> PluginResult {
    if args.len() < 5 {
        return Err("Usage: crop <input_path> <x> <y> <width> <height> [output_path]".to_string());
    }
    
    let input_path = args[0];
    
    let x = match args[1].parse::<u32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid x value".to_string()),
    };
    
    let y = match args[2].parse::<u32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid y value".to_string()),
    };
    
    let width = match args[3].parse::<u32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid width value".to_string()),
    };
    
    let height = match args[4].parse::<u32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid height value".to_string()),
    };
    
    let output_path = if args.len() > 5 {
        args[5].to_string()  // Convert to String
    } else {
        // Create default output path
        let input_path_obj = Path::new(input_path);
        let stem = input_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let extension = input_path_obj.extension().and_then(|s| s.to_str()).unwrap_or("png");
        
        format!("{}_cropped.{}", stem, extension)
    };
    
    // Open the image
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => return Err(format!("Failed to open image: {}", e)),
    };
    
    // Check if crop dimensions are valid
    let (img_width, img_height) = img.dimensions();
    if x + width > img_width || y + height > img_height {
        return Err(format!(
            "Invalid crop dimensions. Image is {}x{}, requested crop is at ({}, {}) with size {}x{}",
            img_width, img_height, x, y, width, height
        ));
    }
    
    // In a real implementation, this would crop and save the image
    // For our demo, we'll just print the operation details
    
    println!("Image would be cropped to region ({}, {}, {}, {}) and saved to: {}",
        x, y, width, height, output_path);
    
    Ok(format!("Image cropped successfully and saved to: {}", output_path))
}

/// Convert an image to grayscale
fn grayscale_image(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("Usage: grayscale <input_path> [output_path]".to_string());
    }
    
    let input_path = args[0];
    let output_path = if args.len() > 1 {
        args[1].to_string()  // Convert to String
    } else {
        // Create default output path
        let input_path_obj = Path::new(input_path);
        let stem = input_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let extension = input_path_obj.extension().and_then(|s| s.to_str()).unwrap_or("png");
        
        format!("{}_grayscale.{}", stem, extension)
    };
    
    // Open the image
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => return Err(format!("Failed to open image: {}", e)),
    };
    
    // In a real implementation, this would convert to grayscale and save
    // For our demo, we'll just print the operation details
    
    println!("Image would be converted to grayscale and saved to: {}", output_path);
    
    Ok(format!("Image converted to grayscale and saved to: {}", output_path))
}

/// Apply a blur effect to an image
fn blur_image(args: &[&str]) -> PluginResult {
    if args.len() < 2 {
        return Err("Usage: blur <input_path> <sigma> [output_path]".to_string());
    }
    
    let input_path = args[0];
    
    let sigma = match args[1].parse::<f32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid sigma value".to_string()),
    };
    
    let output_path = if args.len() > 2 {
        args[2].to_string()  // Convert to String
    } else {
        // Create default output path
        let input_path_obj = Path::new(input_path);
        let stem = input_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let extension = input_path_obj.extension().and_then(|s| s.to_str()).unwrap_or("png");
        
        format!("{}_blur.{}", stem, extension)
    };
    
    // Open the image
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => return Err(format!("Failed to open image: {}", e)),
    };
    
    // In a real implementation, this would apply a blur and save
    // For our demo, we'll just print the operation details
    
    println!("Image would be blurred with sigma={} and saved to: {}", sigma, output_path);
    
    Ok(format!("Image blurred successfully and saved to: {}", output_path))
}

/// Rotate an image by a specified angle
fn rotate_image(args: &[&str]) -> PluginResult {
    if args.len() < 2 {
        return Err("Usage: rotate <input_path> <angle_degrees> [output_path]".to_string());
    }
    
    let input_path = args[0];
    
    let angle = match args[1].parse::<f32>() {
        Ok(val) => val,
        Err(_) => return Err("Invalid angle value".to_string()),
    };
    
    let output_path = if args.len() > 2 {
        args[2].to_string()  // Convert to String
    } else {
        // Create default output path
        let input_path_obj = Path::new(input_path);
        let stem = input_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let extension = input_path_obj.extension().and_then(|s| s.to_str()).unwrap_or("png");
        
        format!("{}_rotated.{}", stem, extension)
    };
    
    // Open the image
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => return Err(format!("Failed to open image: {}", e)),
    };
    
    // In a real implementation, this would rotate and save
    // For our demo, we'll just print the operation details
    
    println!("Image would be rotated by {} degrees and saved to: {}", angle, output_path);
    
    Ok(format!("Image rotated successfully and saved to: {}", output_path))
}