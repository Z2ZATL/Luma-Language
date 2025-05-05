use super::super::ir::IR;

pub fn compile_to_colab(ir: &IR) -> String {
    match ir {
        IR::Load(path, name, lazy) => {
            format!("luma.load_dataset(\"{}\", \"{}\", lazy={})", path, name, if *lazy { "True" } else { "False" })
        }
        IR::Create(model_type) => {
            format!("luma.create_model(\"{}\")", model_type)
        }
        _ => String::new(),
    }
}