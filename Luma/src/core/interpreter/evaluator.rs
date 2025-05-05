use crate::core_parser::ast::{AstNode, Scope};

pub fn evaluate(ast: AstNode, scope: &mut Scope) -> Result<(), String> {
    match ast {
        AstNode::Empty => Ok(()),
        AstNode::Block(nodes) => {
            for node in nodes {
                evaluate(node, scope)?;
            }
            Ok(())
        }
        AstNode::LoadDataset { path, name, lazy } => {
            let name_clone = name.clone();
            scope.insert(name_clone.clone(), AstNode::LoadDataset { path: path.clone(), name: name_clone, lazy });
            println!("Loaded dataset from {} as {} (lazy={})", path, name, lazy);
            Ok(())
        }
        AstNode::CreateModel { model_type } => {
            println!("Created model of type {}", model_type);
            Ok(())
        }
        AstNode::TrainModel { epochs, batch_size, learning_rate } => {
            println!("Training model with epochs={}, batch_size={}, learning_rate={}", epochs, batch_size, learning_rate);
            Ok(())
        }
        AstNode::EvaluateModel { metrics } => {
            println!("Evaluating model with metrics: {:?}", metrics);
            Ok(())
        }
        AstNode::SaveModel { path } => {
            println!("Saved model to {}", path);
            Ok(())
        }
        AstNode::ExecutePlugin { plugin_name, args } => {
            println!("Executing plugin {} with args: {:?}", plugin_name, args);
            Ok(())
        }
    }
}