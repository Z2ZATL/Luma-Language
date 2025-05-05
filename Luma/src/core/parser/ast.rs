use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum AstNode {
    Empty,
    Block(Vec<AstNode>),
    LoadDataset { path: String, name: String, lazy: bool },
    CreateModel { model_type: String },
    TrainModel { epochs: i32, batch_size: i32, learning_rate: f64 },
    EvaluateModel { metrics: Vec<String> },
    SaveModel { path: String },
    ExecutePlugin { plugin_name: String, args: Vec<String> },
}

pub struct Scope {
    variables: HashMap<String, AstNode>,
}

impl Scope {
    pub fn new() -> Self {
        Scope {
            variables: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, node: AstNode) {
        self.variables.insert(name, node);
    }

    pub fn get(&self, name: &str) -> Option<&AstNode> {
        self.variables.get(name)
    }
}