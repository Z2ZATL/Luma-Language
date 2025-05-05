use crate::plugins::registry;

pub struct ImageProcessingPlugin;

impl ImageProcessingPlugin {
    pub fn new() -> Self {
        ImageProcessingPlugin
    }
}

impl registry::Plugin for ImageProcessingPlugin {
    fn execute(&self, args: Vec<String>) -> Result<String, String> {
        if args.is_empty() {
            Err("No arguments provided for ImageProcessingPlugin".to_string())
        } else {
            Ok(format!("ImageProcessingPlugin processed with args: {:?}", args))
        }
    }
}