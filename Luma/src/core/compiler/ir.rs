#[derive(Debug)]
pub enum IR {
    Load(String, String, bool),
    Create(String),
    Train(i32, i32, f64),
    Evaluate(Vec<String>),
    Save(String),
}

pub fn ast_to_ir(ast: AstNode) -> IR {
    match ast {
        AstNode::LoadDataset { path, name, lazy } => IR::Load(path, name, lazy),
        AstNode::CreateModel { model_type } => IR::Create(model_type),
        AstNode::TrainModel { epochs, batch_size, learning_rate } => IR::Train(epochs, batch_size, learning_rate),
        AstNode::EvaluateModel { metrics } => IR::Evaluate(metrics),
        AstNode::SaveModel { path } => IR::Save(path),
    }
}