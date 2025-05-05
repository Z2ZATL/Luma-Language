use std::env;
use std::fs;
use luma::core::interpreter::pipeline_executor;
use luma::repl;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        if args[1] == "--repl" {
            repl::repl::start_repl();
        } else {
            // Assume the argument is a file path
            let file_path = &args[1];
            match fs::read_to_string(file_path) {
                Ok(contents) => {
                    if let Err(e) = pipeline_executor::execute_pipeline(&contents) {
                        eprintln!("Error executing file: {}", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error reading file '{}': {}", file_path, e);
                    std::process::exit(1);
                }
            }
        }
    } else {
        println!("Usage:");
        println!("  cargo run -- <file.luma>   - Run a Luma script");
        println!("  cargo run -- --repl        - Start Luma REPL (interactive mode)");
    }
}