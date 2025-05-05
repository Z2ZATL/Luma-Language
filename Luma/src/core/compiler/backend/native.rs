use super::super::ir::IR;

pub fn compile_to_native(ir: &IR) -> String {
    match ir {
        IR::Load(path, name, lazy) => {
            format!("luma_load_dataset(\"{}\", \"{}\", {});", path, name, if *lazy { 1 } else { 0 })
        }
        IR::Create(model_type) => {
            format!("luma_create_model(\"{}\");", model_type)
        }
        _ => String::new(),
    }
}