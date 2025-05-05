use crate::plugins::registry;

pub struct MultiModalPlugin;

impl MultiModalPlugin {
    pub fn new() -> Self {
        MultiModalPlugin
    }
}

impl registry::Plugin for MultiModalPlugin {
    fn execute(&self, args: Vec<String>) -> Result<String, String> {
        if args.is_empty() {
            Err("No arguments provided for MultiModalPlugin".to_string())
        } else {
            Ok(format!("MultiModalPlugin processed with args: {:?}", args))
        }
    }
}