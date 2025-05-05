use crate::core_parser::ast::AstNode;
use crate::core_parser::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<AstNode, String> {
        let mut statements = Vec::new();
        while self.pos < self.tokens.len() {
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            } else {
                break;
            }
        }
        if statements.is_empty() {
            Ok(AstNode::Empty)
        } else if statements.len() == 1 {
            Ok(statements.into_iter().next().unwrap())
        } else {
            Ok(AstNode::Block(statements))
        }
    }

    fn parse_statement(&mut self) -> Result<Option<AstNode>, String> {
        if self.pos >= self.tokens.len() {
            return Ok(None);
        }

        match self.current_token() {
            Some(Token::Load) => self.parse_load(),
            Some(Token::Create) => self.parse_create(),
            Some(Token::Train) => self.parse_train(),
            Some(Token::Evaluate) => self.parse_evaluate(),
            Some(Token::Save) => self.parse_save(),
            Some(Token::Execute) => self.parse_execute(),
            _ => {
                self.pos += 1;
                Ok(None)
            }
        }
    }

    fn parse_load(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "load"
        if !self.expect(Token::Dataset)? {
            return Err("Expected 'dataset' after 'load'".to_string());
        }
        self.pos += 1; // Skip "dataset"

        let path = match self.current_token() {
            Some(Token::StringLiteral(s)) => s.clone(),
            _ => return Err("Expected string literal for dataset path".to_string()),
        };
        self.pos += 1; // Skip path

        if !self.expect(Token::As)? {
            return Err("Expected 'as' after dataset path".to_string());
        }
        self.pos += 1; // Skip "as"

        let name = match self.current_token() {
            Some(Token::Identifier(id)) => id.clone(),
            _ => return Err("Expected identifier for dataset name".to_string()),
        };
        self.pos += 1; // Skip name

        let mut lazy = false;
        if self.expect(Token::Lazy)? {
            self.pos += 1; // Skip "lazy"
            if !self.expect(Token::Equals)? {
                return Err("Expected '=' after 'lazy'".to_string());
            }
            self.pos += 1; // Skip "="
            match self.current_token() {
                Some(Token::True) => lazy = true,
                Some(Token::False) => lazy = false,
                _ => return Err("Expected 'true' or 'false' for lazy".to_string()),
            }
            self.pos += 1; // Skip true/false
        }

        Ok(Some(AstNode::LoadDataset { path, name, lazy }))
    }

    fn parse_create(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "create"
        if !self.expect(Token::Model)? {
            return Err("Expected 'model' after 'create'".to_string());
        }
        self.pos += 1; // Skip "model"

        let model_type = match self.current_token() {
            Some(Token::Identifier(id)) => id.clone(),
            _ => return Err("Expected model type".to_string()),
        };
        self.pos += 1; // Skip model_type

        Ok(Some(AstNode::CreateModel { model_type }))
    }

    fn parse_train(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "train"

        let mut epochs = 0;
        let mut batch_size = 0;
        let mut learning_rate = 0.0;

        while self.pos < self.tokens.len() {
            match self.current_token() {
                Some(Token::Epochs) => {
                    self.pos += 1; // Skip "epochs"
                    if !self.expect(Token::Equals)? {
                        return Err("Expected '=' after 'epochs'".to_string());
                    }
                    self.pos += 1; // Skip "="
                    epochs = match self.current_token() {
                        Some(Token::Number(n)) => *n as i32,
                        _ => return Err("Expected number for epochs".to_string()),
                    };
                    self.pos += 1; // Skip number
                }
                Some(Token::BatchSize) => {
                    self.pos += 1; // Skip "batch_size"
                    if !self.expect(Token::Equals)? {
                        return Err("Expected '=' after 'batch_size'".to_string());
                    }
                    self.pos += 1; // Skip "="
                    batch_size = match self.current_token() {
                        Some(Token::Number(n)) => *n as i32,
                        _ => return Err("Expected number for batch_size".to_string()),
                    };
                    self.pos += 1; // Skip number
                }
                Some(Token::LearningRate) => {
                    self.pos += 1; // Skip "learning_rate"
                    if !self.expect(Token::Equals)? {
                        return Err("Expected '=' after 'learning_rate'".to_string());
                    }
                    self.pos += 1; // Skip "="
                    learning_rate = match self.current_token() {
                        Some(Token::Float(f)) => *f,
                        _ => return Err("Expected float for learning_rate".to_string()),
                    };
                    self.pos += 1; // Skip float
                }
                _ => break,
            }
        }

        Ok(Some(AstNode::TrainModel { epochs, batch_size, learning_rate }))
    }

    fn parse_evaluate(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "evaluate"

        let mut metrics = Vec::new();
        while self.pos < self.tokens.len() {
            match self.current_token() {
                Some(Token::Identifier(metric)) => {
                    metrics.push(metric.clone());
                    self.pos += 1;
                }
                _ => break,
            }
        }

        Ok(Some(AstNode::EvaluateModel { metrics }))
    }

    fn parse_save(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "save"

        let path = match self.current_token() {
            Some(Token::StringLiteral(s)) => s.clone(),
            _ => return Err("Expected string literal for save path".to_string()),
        };
        self.pos += 1; // Skip path

        Ok(Some(AstNode::SaveModel { path }))
    }

    fn parse_execute(&mut self) -> Result<Option<AstNode>, String> {
        self.pos += 1; // Skip "execute"

        if !self.expect(Token::Plugin)? {
            return Err("Expected 'plugin' after 'execute'".to_string());
        }
        self.pos += 1; // Skip "plugin"

        let plugin_name = match self.current_token() {
            Some(Token::Identifier(id)) => id.clone(),
            _ => return Err("Expected plugin name".to_string()),
        };
        self.pos += 1; // Skip plugin_name

        let mut args = Vec::new();
        while self.pos < self.tokens.len() {
            match self.current_token() {
                Some(Token::StringLiteral(s)) => {
                    args.push(s.clone());
                    self.pos += 1;
                }
                Some(Token::Number(n)) => {
                    args.push(n.to_string());
                    self.pos += 1;
                }
                Some(Token::Float(f)) => {
                    args.push(f.to_string());
                    self.pos += 1;
                }
                _ => break,
            }
        }

        Ok(Some(AstNode::ExecutePlugin { plugin_name, args }))
    }

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn expect(&self, expected: Token) -> Result<bool, String> {
        match self.current_token() {
            Some(token) if *token == expected => Ok(true),
            Some(token) => Err(format!("Expected token {:?}, found {:?}", expected, token)),
            None => Err(format!("Expected token {:?}, found end of input", expected)),
        }
    }
}