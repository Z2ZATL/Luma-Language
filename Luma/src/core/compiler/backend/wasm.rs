use super::super::ir::IR;

pub fn compile_to_wasm(ir: &IR) -> String {
    match ir {
        IR::Load(path, name, lazy) => {
            format!("Module.load_dataset(\"{}\", \"{}\", {});", path, name, if *lazy { "true" } else { "false" })
        }
        IR::Create(model_type) => {
            format!("Module.create_model(\"{}\");", model_type)
        }
        _ => String::new(),
    }
}