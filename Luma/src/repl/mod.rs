pub mod repl {
    use crate::core_interpreter::pipeline_executor;
    use std::io::{self, Write};

    pub fn start_repl() {
        println!("🌟 Luma REPL v0.1.0 🌟");
        println!("Type 'exit' to quit. Enter Luma commands to interact.");
        println!("Example: load dataset \"iris.csv\" as iris lazy=True");

        let mut input = String::new();
        loop {
            print!("luma> ");
            io::stdout().flush().unwrap();

            input.clear();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.is_empty() {
                continue;
            }
            if input.eq_ignore_ascii_case("exit") {
                println!("👋 Exiting Luma REPL. Goodbye!");
                break;
            }

            match pipeline_executor::execute_pipeline(input) {
                Ok(()) => println!("✅ Command executed successfully."),
                Err(e) => eprintln!("❌ Error: {}", e),
            }
        }
    }
}