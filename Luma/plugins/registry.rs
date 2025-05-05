use crate::plugins::community::custom_nlp_module::NLPPlugin;
use crate::plugins::community::image_processing::ImageProcessingPlugin;
use crate::plugins::community::multi_modal_support::MultiModalPlugin;

pub trait Plugin {
    fn execute(&self, args: Vec<String>) -> Result<String, String>;
}

pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginManager {
    pub fn new() -> Self {
        let mut manager = PluginManager { plugins: Vec::new() };
        manager.plugins.push(Box::new(MultiModalPlugin::new()));
        manager.plugins.push(Box::new(NLPPlugin::new()));
        manager.plugins.push(Box::new(ImageProcessingPlugin::new()));
        manager
    }

    pub fn execute_plugin(&self, plugin_name: &str, args: Vec<String>) -> Result<String, String> {
        for plugin in &self.plugins {
            if plugin_name == std::any::type_name::<Self>().split("::").last().unwrap() {
                return plugin.execute(args);
            }
        }
        Err(format!("Plugin {} not found", plugin_name))
    }
}