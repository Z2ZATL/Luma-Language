use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use chrono::Local;
use std::fmt;
use lazy_static::lazy_static;

/// Logging levels similar to those in standard logging libraries
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Fatal = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Fatal => write!(f, "FATAL"),
        }
    }
}

impl LogLevel {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warning" | "warn" => Ok(LogLevel::Warning),
            "error" => Ok(LogLevel::Error),
            "fatal" | "critical" => Ok(LogLevel::Fatal),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogConfig {
    pub level: LogLevel,
    pub log_file: String,
    pub console_output: bool,
    pub include_timestamp: bool,
    pub include_source: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig {
            level: LogLevel::Info,
            log_file: "luma.log".to_string(),
            console_output: true,
            include_timestamp: true,
            include_source: true,
        }
    }
}

lazy_static! {
    static ref LOGGER: Mutex<Logger> = Mutex::new(Logger::new());
}

pub struct Logger {
    config: LogConfig,
}

impl Logger {
    fn new() -> Self {
        Logger {
            config: LogConfig::default(),
        }
    }

    fn format_log(&self, level: LogLevel, message: &str, source: Option<&str>) -> String {
        let mut log_parts = Vec::new();
        
        if self.config.include_timestamp {
            let now = Local::now();
            log_parts.push(now.format("%Y-%m-%d %H:%M:%S%.3f").to_string());
        }
        
        log_parts.push(format!("[{}]", level));
        
        if self.config.include_source && source.is_some() {
            log_parts.push(format!("[{}]", source.unwrap()));
        }
        
        log_parts.push(message.to_string());
        
        log_parts.join(" ")
    }

    fn write_log(&self, formatted_message: &str) -> Result<(), String> {
        // Ensure the directory exists
        if let Some(parent) = Path::new(&self.config.log_file).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create log directory: {}", e))?;
            }
        }
        
        // Write to file
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.config.log_file)
            .map_err(|e| format!("Failed to open log file: {}", e))?;
            
        writeln!(file, "{}", formatted_message)
            .map_err(|e| format!("Failed to write to log file: {}", e))?;
        
        // Also write to console if enabled
        if self.config.console_output {
            println!("{}", formatted_message);
        }
        
        Ok(())
    }
}

/// Configure the global logger
pub fn configure_logger(config: LogConfig) -> Result<(), String> {
    let mut logger = LOGGER.lock().map_err(|_| "Failed to acquire logger lock")?;
    logger.config = config;
    Ok(())
}

/// Get the current log configuration
pub fn get_log_config() -> Result<LogConfig, String> {
    let logger = LOGGER.lock().map_err(|_| "Failed to acquire logger lock")?;
    Ok(logger.config.clone())
}

/// Set the log level
pub fn set_log_level(level: LogLevel) -> Result<(), String> {
    let mut logger = LOGGER.lock().map_err(|_| "Failed to acquire logger lock")?;
    logger.config.level = level;
    Ok(())
}

/// Log a message with the specified level
pub fn log(level: LogLevel, message: &str, source: Option<&str>) -> Result<(), String> {
    let logger = LOGGER.lock().map_err(|_| "Failed to acquire logger lock")?;
    
    // Check if we should log this message based on level
    if level >= logger.config.level {
        let formatted = logger.format_log(level, message, source);
        logger.write_log(&formatted)?;
    }
    
    Ok(())
}

/// Convenience functions for different log levels
pub fn trace(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Trace, message, source)
}

pub fn debug(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Debug, message, source)
}

pub fn info(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Info, message, source)
}

pub fn warn(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Warning, message, source)
}

pub fn error(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Error, message, source)
}

pub fn fatal(message: &str, source: Option<&str>) -> Result<(), String> {
    log(LogLevel::Fatal, message, source)
}

/// Simple legacy function for backward compatibility
pub fn log_message(message: &str) -> Result<(), String> {
    info(message, None)
}

#[no_mangle]
pub extern "C" fn luma_log_message(message: *const std::os::raw::c_char) -> i32 {
    if message.is_null() {
        return -1;
    }
    
    let msg_str = unsafe { 
        match std::ffi::CStr::from_ptr(message).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    match log_message(msg_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Macro for easier logging with file and line information
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::utilities::logging::info(&format!($($arg)*), Some(&format!("{}:{}", file!(), line!())))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::utilities::logging::error(&format!($($arg)*), Some(&format!("{}:{}", file!(), line!())))
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::utilities::logging::debug(&format!($($arg)*), Some(&format!("{}:{}", file!(), line!())))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    
    #[test]
    fn test_log_levels() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
        
        let level = LogLevel::from_str("info").unwrap();
        assert_eq!(level, LogLevel::Info);
        
        let level_str = LogLevel::Warning.to_string();
        assert_eq!(level_str, "WARN");
    }
    
    #[test]
    fn test_logging() {
        let test_log = "test_logging.log";
        
        // Set up a test configuration
        let config = LogConfig {
            level: LogLevel::Debug,
            log_file: test_log.to_string(),
            console_output: false,
            include_timestamp: true,
            include_source: true,
        };
        
        configure_logger(config).unwrap();
        
        // Test various log levels
        debug("This is a debug message", Some("test_logging")).unwrap();
        info("This is an info message", Some("test_logging")).unwrap();
        warn("This is a warning message", None).unwrap();
        
        // Verify log file exists
        assert!(Path::new(test_log).exists());
        
        // Clean up
        fs::remove_file(test_log).unwrap();
    }
}