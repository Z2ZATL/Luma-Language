#[derive(Debug, PartialEq)]
pub enum AstNode {
    LoadDataset {
        path: String,
        name: String,
        lazy: bool,
    },
    CreateModel {
        model_type: String,
    },
    TrainModel {
        epochs: i32,
        batch_size: i32,
        learning_rate: f64,
    },
    EvaluateModel {
        metrics: Vec<String>,
    },
    SaveModel {
        path: String,
    },
}