//! Plugin Registry for Luma
//!
//! This module handles the registration, discovery, and management of plugins
//! in the Luma ecosystem.

use crate::plugins::{Plugin, list_plugins};
use std::path::Path;

/// Scan a directory for Luma plugins
pub fn scan_directory(dir_path: &str) -> Result<Vec<String>, String> {
    let path = Path::new(dir_path);
    if !path.exists() || !path.is_dir() {
        return Err(format!("Invalid plugin directory: {}", dir_path));
    }

    let mut found_plugins = Vec::new();
    
    println!("Scanning for plugins in: {}", dir_path);
    
    // In a real implementation, this would look for .so, .dll, or .dylib files
    // and attempt to load them as dynamic libraries containing Luma plugins.
    // For our demo, we'll just return a success message.
    
    found_plugins.push(format!("Scanned directory: {}", dir_path));
    
    Ok(found_plugins)
}

/// Get metadata for a specific plugin
pub fn get_plugin_metadata(plugin_id: &str) -> Option<Plugin> {
    let plugins = list_plugins();
    plugins.into_iter().find(|p| p.id == plugin_id)
}

/// Print detailed information about all registered plugins
pub fn print_plugins_info() {
    let plugins = list_plugins();
    
    if plugins.is_empty() {
        println!("No plugins registered.");
        return;
    }
    
    println!("Registered Luma Plugins:");
    println!("{:<15} {:<25} {:<10} {:<15} {:<40}", "ID", "Name", "Version", "Status", "Description");
    println!("{}", "-".repeat(100));
    
    for plugin in plugins {
        println!("{:<15} {:<25} {:<10} {:<15} {:<40}",
            plugin.id,
            plugin.name,
            plugin.version,
            if plugin.enabled { "Enabled" } else { "Disabled" },
            plugin.description.unwrap_or_else(|| "No description".to_string())
        );
    }
}