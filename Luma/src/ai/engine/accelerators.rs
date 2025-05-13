use std::sync::RwLock;
use lazy_static::lazy_static;
use std::fmt;

/// Represents the hardware acceleration device type
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    CPU,
    CUDA,
    OpenCL,
    Metal,
    TPU,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::CUDA => write!(f, "CUDA"),
            DeviceType::OpenCL => write!(f, "OpenCL"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::TPU => write!(f, "TPU"),
        }
    }
}

impl DeviceType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(DeviceType::CPU),
            "cuda" | "gpu" => Ok(DeviceType::CUDA),
            "opencl" => Ok(DeviceType::OpenCL),
            "metal" => Ok(DeviceType::Metal),
            "tpu" => Ok(DeviceType::TPU),
            _ => Err(format!("Unsupported device type: {}", s)),
        }
    }
}

/// Properties specific to each device type
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub memory_mb: usize,
    pub compute_capability: Option<String>,
}

/// Represents a hardware acceleration device
#[derive(Debug, Clone)]
pub struct Accelerator {
    pub device_type: DeviceType,
    pub device_id: usize,
    pub properties: DeviceProperties,
}

impl Accelerator {
    pub fn new(device_type: DeviceType, device_id: usize) -> Self {
        // Default properties based on device type
        let properties = match device_type {
            DeviceType::CPU => DeviceProperties {
                name: "CPU".to_string(),
                memory_mb: get_system_memory_mb(),
                compute_capability: None,
            },
            DeviceType::CUDA => DeviceProperties {
                name: format!("CUDA Device {}", device_id),
                memory_mb: 0, // Would be populated in a real implementation
                compute_capability: Some("Unknown".to_string()),
            },
            _ => DeviceProperties {
                name: format!("{} Device {}", device_type, device_id),
                memory_mb: 0,
                compute_capability: None,
            },
        };

        Accelerator {
            device_type,
            device_id,
            properties,
        }
    }

    pub fn get_info(&self) -> String {
        format!(
            "Device: {} (ID: {})\nType: {}\nMemory: {} MB",
            self.properties.name, self.device_id, self.device_type, self.properties.memory_mb
        )
    }
}

fn get_system_memory_mb() -> usize {
    // Placeholder - would use system calls to determine actual memory
    // For simplicity, returning a fixed value
    16384 // 16 GB
}

lazy_static! {
    static ref ACCELERATOR: RwLock<Option<Accelerator>> = RwLock::new(None);
}

/// List available devices for machine learning computation
pub fn list_available_devices() -> Vec<(DeviceType, usize)> {
    // In a real implementation, this would query the system for available devices
    // For now, just return CPU as always available
    vec![(DeviceType::CPU, 0)]
}

/// Set the active accelerator device
pub fn set_accelerator(device: &str) -> Result<(), String> {
    let device_type = DeviceType::from_str(device)?;
    
    // Check if the requested device is available
    let available_devices = list_available_devices();
    let device_id = available_devices.iter()
        .find(|(d_type, _)| *d_type == device_type)
        .map(|(_, id)| *id)
        .ok_or_else(|| format!("Device type {} is not available on this system", device_type))?;
    
    let accelerator = Accelerator::new(device_type, device_id);
    
    let mut accel = ACCELERATOR.write().map_err(|_| "Failed to acquire write lock")?;
    *accel = Some(accelerator);
    
    Ok(())
}

/// Get the current accelerator if set
pub fn get_accelerator() -> Option<Accelerator> {
    match ACCELERATOR.read() {
        Ok(accel) => accel.clone(),
        Err(_) => {
            eprintln!("Failed to acquire read lock for accelerator");
            None
        }
    }
}

/// Clear the current accelerator setting
pub fn clear_accelerator() -> Result<(), String> {
    let mut accel = ACCELERATOR.write().map_err(|_| "Failed to acquire write lock")?;
    *accel = None;
    Ok(())
}

/// Get information about the current accelerator
pub fn get_accelerator_info() -> String {
    match get_accelerator() {
        Some(accel) => accel.get_info(),
        None => "No accelerator is currently set".to_string(),
    }
}