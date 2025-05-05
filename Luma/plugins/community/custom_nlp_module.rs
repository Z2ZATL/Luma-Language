use crate::plugins::registry;

pub struct NLPPlugin;

impl NLPPlugin {
    pub fn new() -> Self {
        NLPPlugin
    }
}

impl registry::Plugin for NLPPlugin {
    fn execute(&self, args: Vec<String>) -> Result<String, String> {
        if args.is_empty() {
            Err("No arguments provided for NLPPlugin".to_string())
        } else {
            Ok(format!("NLPPlugin processed with args: {:?}", args))
        }
    }
}