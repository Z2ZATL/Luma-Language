use crate::core_parser::ast::AstNode;

#[derive(Debug)]
pub enum IR {
    Load(String, String, bool),
    Create(String),
    Train(i32, i32, f64),
    Evaluate(Vec<String>),
    Save(String),
    Execute(String, Vec<String>),
    NoOp,
}

pub fn generate_ir(ast: AstNode) -> IR {
    match ast {
        AstNode::LoadDataset { path, name, lazy } => IR::Load(path, name, lazy),
        AstNode::CreateModel { model_type } => IR::Create(model_type),
        AstNode::TrainModel { epochs, batch_size, learning_rate } => {
            IR::Train(epochs, batch_size, learning_rate)
        }
        AstNode::EvaluateModel { metrics } => IR::Evaluate(metrics),
        AstNode::SaveModel { path } => IR::Save(path),
        AstNode::ExecutePlugin { plugin_name, args } => IR::Execute(plugin_name, args),
        AstNode::Empty => IR::NoOp,
        AstNode::Block(_) => IR::NoOp, // Simplified for now, can be expanded later
    }
}