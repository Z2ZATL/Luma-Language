//! Luma Plugin System
//! 
//! This module provides a framework for extending Luma with custom plugins.
//! Plugins can contribute new AI/ML capabilities, data processing, integrations,
//! or other specialized functionality.

pub mod registry;
pub mod nlp;
pub mod image_processing;
pub mod multi_modal;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;

/// Represents a registered Luma plugin
#[derive(Clone)]
pub struct Plugin {
    /// Unique plugin ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Plugin version (semantic versioning)
    pub version: String,
    /// Optional plugin description
    pub description: Option<String>,
    /// Plugin author information
    pub author: Option<String>,
    /// Available plugin commands
    pub commands: Vec<String>,
    /// Indicates if the plugin is currently enabled
    pub enabled: bool,
}

/// Plugin execution result type
pub type PluginResult = Result<String, String>;

/// Plugin execution function type
pub type PluginExecFn = fn(args: &[&str]) -> PluginResult;

lazy_static! {
    /// Registry of available plugins
    static ref PLUGINS: RwLock<HashMap<String, Plugin>> = RwLock::new(HashMap::new());

    /// Registry of plugin execution functions
    static ref PLUGIN_FUNCTIONS: RwLock<HashMap<String, PluginExecFn>> = RwLock::new(HashMap::new());
}

/// Register a new plugin in the Luma system
pub fn register_plugin(
    id: &str,
    name: &str,
    version: &str,
    description: Option<&str>,
    author: Option<&str>,
    commands: Vec<String>,
    exec_fn: PluginExecFn,
) -> Result<(), String> {
    // Create the plugin entry
    let plugin = Plugin {
        id: id.to_string(),
        name: name.to_string(),
        version: version.to_string(),
        description: description.map(|s| s.to_string()),
        author: author.map(|s| s.to_string()),
        commands: commands.clone(),
        enabled: true,
    };

    // Lock the plugins registry for writing
    let mut plugins = match PLUGINS.write() {
        Ok(guard) => guard,
        Err(err) => return Err(format!("Lock error: {}", err)),
    };

    // Check for existing plugin with the same ID
    if plugins.contains_key(id) {
        return Err(format!("Plugin with ID '{}' already exists", id));
    }

    // Add plugin to registry
    plugins.insert(id.to_string(), plugin);

    // Add execution function to registry
    let mut functions = match PLUGIN_FUNCTIONS.write() {
        Ok(guard) => guard,
        Err(err) => return Err(format!("Lock error: {}", err)),
    };

    functions.insert(id.to_string(), exec_fn);

    println!("Registered plugin '{}' ({})", name, id);
    Ok(())
}

/// Execute a plugin command
pub fn execute_plugin(plugin_id: &str, args: &[&str]) -> PluginResult {
    // Check if plugin exists and is enabled
    let plugins = match PLUGINS.read() {
        Ok(guard) => guard,
        Err(err) => return Err(format!("Lock error: {}", err)),
    };

    if !plugins.contains_key(plugin_id) {
        return Err(format!("Plugin '{}' not found", plugin_id));
    }

    let plugin = &plugins[plugin_id];
    if !plugin.enabled {
        return Err(format!("Plugin '{}' is disabled", plugin_id));
    }

    // Get execution function
    let functions = match PLUGIN_FUNCTIONS.read() {
        Ok(guard) => guard,
        Err(err) => return Err(format!("Lock error: {}", err)),
    };

    if let Some(exec_fn) = functions.get(plugin_id) {
        // Execute plugin function
        exec_fn(args)
    } else {
        Err(format!("No execution function found for plugin '{}'", plugin_id))
    }
}

/// List all registered plugins
pub fn list_plugins() -> Vec<Plugin> {
    let plugins = match PLUGINS.read() {
        Ok(guard) => guard,
        Err(_) => return Vec::new(),
    };

    plugins.values().cloned().collect()
}

/// Enable or disable a plugin
pub fn set_plugin_enabled(plugin_id: &str, enabled: bool) -> Result<(), String> {
    let mut plugins = match PLUGINS.write() {
        Ok(guard) => guard,
        Err(err) => return Err(format!("Lock error: {}", err)),
    };

    if let Some(plugin) = plugins.get_mut(plugin_id) {
        plugin.enabled = enabled;
        Ok(())
    } else {
        Err(format!("Plugin '{}' not found", plugin_id))
    }
}

/// Initialize the plugin system and register built-in plugins
pub fn init_plugin_system() {
    // Register built-in plugins
    let _ = nlp::register_nlp_plugin();
    let _ = image_processing::register_image_plugin();
    let _ = multi_modal::register_multi_modal_plugin();
    
    // Community plugin support has been temporarily disabled
    // let _ = community::load_community_plugins();
}