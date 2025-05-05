use crate::core_parser::ast::AstNode;

pub fn generate_code(ast: AstNode) -> String {
    match ast {
        AstNode::Empty => "NOOP".to_string(),
        AstNode::Block(nodes) => nodes.into_iter().map(generate_code).collect::<Vec<String>>().join("; "),
        AstNode::LoadDataset { path, name, lazy } => format!("LOAD DATASET {} AS {} LAZY={}", path, name, lazy),
        AstNode::CreateModel { model_type } => format!("CREATE MODEL {}", model_type),
        AstNode::TrainModel { epochs, batch_size, learning_rate } => {
            format!("TRAIN MODEL epochs={} batch_size={} learning_rate={}", epochs, batch_size, learning_rate)
        }
        AstNode::EvaluateModel { metrics } => format!("EVALUATE MODEL metrics={:?}", metrics),
        AstNode::SaveModel { path } => format!("SAVE MODEL {}", path),
        AstNode::ExecutePlugin { plugin_name, args } => format!("EXECUTE PLUGIN {} {:?}", plugin_name, args),
    }
}