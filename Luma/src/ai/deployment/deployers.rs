use std::fs::File;
use std::io::Write;

pub fn deploy_model(model_path: &str, target: &str) -> Result<(), String> {
    let mut file = File::create(target).map_err(|e| e.to_string())?;
    file.write_all(model_path.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}