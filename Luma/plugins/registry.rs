use std::collections::HashMap;
use std::sync::Arc;
use std::cell::RefCell;

pub struct PluginRegistry {
    plugins: Arc<RefCell<HashMap<String, Box<dyn Plugin>>>>,
}

pub trait Plugin {
    fn name(&self) -> &str;
    fn execute(&self, command: &str, args: Vec<&str>) -> Result<String, String>;
}

impl PluginRegistry {
    pub fn new() -> Self {
        PluginRegistry {
            plugins: Arc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn register(&self, plugin: Box<dyn Plugin>) {
        let name = plugin.name().to_string();
        let mut plugins = self.plugins.borrow_mut();
        plugins.insert(name, plugin);
    }

    pub fn execute(&self, plugin_name: &str, command: &str, args: Vec<&str>) -> Result<String, String> {
        let plugins = self.plugins.borrow();
        let plugin = plugins.get(plugin_name)
            .ok_or_else(|| format!("Plugin '{}' not found", plugin_name))?;
        plugin.execute(command, args)
    }
}

// Implement plugins as part of registry
struct MultiModalPluginWrapper(MultiModalPlugin);

impl Plugin for MultiModalPluginWrapper {
    fn name(&self) -> &str {
        "multi_modal"
    }

    fn execute(&self, command: &str, args: Vec<&str>) -> Result<String, String> {
        if command == "add_text" && args.len() >= 2 {
            let id: i32 = args[0].parse().map_err(|_| "Invalid ID".to_string())?;
            let text_data = args[1..].to_vec();
            self.0.add_text_data(id, text_data)?;
            Ok("Text data added".to_string())
        } else {
            Err("Invalid command or arguments".to_string())
        }
    }
}

struct NLPPluginWrapper(NLPPlugin);

impl Plugin for NLPPluginWrapper {
    fn name(&self) -> &str {
        "nlp"
    }

    fn execute(&self, command: &str, args: Vec<&str>) -> Result<String, String> {
        if command == "tokenize" && args.len() == 1 {
            let tokens = self.0.tokenize(args[0]);
            Ok(tokens.join(","))
        } else if command == "sentiment" && args.len() == 1 {
            let score = self.0.sentiment_analysis(args[0]);
            Ok(score.to_string())
        } else {
            Err("Invalid command or arguments".to_string())
        }
    }
}

#[no_mangle]
pub extern "C" fn luma_register_plugin(registry: *mut PluginRegistry, plugin_name: *const std::os::raw::c_char) -> i32 {
    if registry.is_null() || plugin_name.is_null() {
        return -1;
    }
    let name_str = unsafe { std::ffi::CStr::from_ptr(plugin_name).to_str() };
    if name_str.is_err() {
        return -1;
    }
    let plugin_name = name_str.unwrap();
    let registry = unsafe { &mut *registry };
    match plugin_name {
        "multi_modal" => {
            let plugin = Box::new(MultiModalPluginWrapper(MultiModalPlugin::new()));
            registry.register(plugin);
        }
        "nlp" => {
            let plugin = Box::new(NLPPluginWrapper(NLPPlugin::new()));
            registry.register(plugin);
        }
        _ => return -1,
    }
    0
}

#[no_mangle]
pub extern "C" fn luma_execute_plugin(registry: *const PluginRegistry, plugin_name: *const std::os::raw::c_char, command: *const std::os::raw::c_char, args: *const *const std::os::raw::c_char, args_len: i32) -> *mut std::os::raw::c_char {
    if registry.is_null() || plugin_name.is_null() || command.is_null() || args.is_null() || args_len < 0 {
        return std::ptr::null_mut();
    }
    let plugin_name_str = unsafe { std::ffi::CStr::from_ptr(plugin_name).to_str() };
    let command_str = unsafe { std::ffi::CStr::from_ptr(command).to_str() };
    if plugin_name_str.is_err() || command_str.is_err() {
        return std::ptr::null_mut();
    }
    let mut args_vec = Vec::new();
    unsafe {
        for i in 0..args_len as isize {
            let arg = *args.offset(i);
            if !arg.is_null() {
                if let Ok(s) = std::ffi::CStr::from_ptr(arg).to_str() {
                    args_vec.push(s);
                }
            }
        }
    }
    let registry = unsafe { &*registry };
    let result = registry.execute(plugin_name_str.unwrap(), command_str.unwrap(), args_vec);
    match result {
        Ok(s) => {
            let c_str = std::ffi::CString::new(s).unwrap();
            c_str.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn luma_create_plugin_registry() -> *mut PluginRegistry {
    let registry = Box::new(PluginRegistry::new());
    Box::into_raw(registry)
}

#[no_mangle]
pub extern "C" fn luma_free_plugin_registry(registry: *mut PluginRegistry) {
    if !registry.is_null() {
        unsafe { Box::from_raw(registry); }
    }
}