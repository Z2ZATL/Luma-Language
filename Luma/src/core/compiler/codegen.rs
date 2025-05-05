use super::super::parser::ast::AstNode;

pub fn generate_code(ast: AstNode) -> String {
    match ast {
        AstNode::LoadDataset { path, name, lazy } => {
            format!("luma_load_dataset(\"{}\", \"{}\", {});", path, name, if lazy { 1 } else { 0 })
        }
        AstNode::CreateModel { model_type } => {
            format!("luma_create_model(\"{}\");", model_type)
        }
        AstNode::TrainModel { epochs, batch_size, learning_rate } => {
            format!("luma_train({}, {}, {}, {});", 1, epochs, batch_size, learning_rate) // model_id = 1 (placeholder)
        }
        AstNode::EvaluateModel { metrics } => {
            format!("luma_evaluate({}, \"{}\");", 1, metrics.join(",")) // model_id = 1 (placeholder)
        }
        AstNode::SaveModel { path } => {
            format!("luma_save_model({}, \"{}\");", 1, path) // model_id = 1 (placeholder)
        }
    }
}