use super::lexer::{Lexer, Token};
use super::ast::AstNode;

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut parser = Parser {
            lexer,
            current_token: Token::EOF,
        };
        parser.next_token();
        parser
    }

    fn next_token(&mut self) {
        self.current_token = self.lexer.next_token();
    }

    pub fn parse(&mut self) -> Result<AstNode, String> {
        match &self.current_token {
            Token::Load => self.parse_load(),
            Token::Create => self.parse_create(),
            Token::Train => self.parse_train(),
            Token::Evaluate => self.parse_evaluate(),
            Token::Save => self.parse_save(),
            _ => Err(format!("Unexpected token: {:?}", self.current_token)),
        }
    }

    fn parse_load(&mut self) -> Result<AstNode, String> {
        self.next_token();
        if self.current_token != Token::Dataset {
            return Err("Expected 'dataset' after 'load'".to_string());
        }
        self.next_token();
        let path = match &self.current_token {
            Token::String(s) => s.clone(),
            _ => return Err("Expected string path after 'dataset'".to_string()),
        };
        self.next_token();
        if self.current_token != Token::As {
            return Err("Expected 'as' after path".to_string());
        }
        self.next_token();
        let name = match &self.current_token {
            Token::Identifier(id) => id.clone(),
            _ => return Err("Expected identifier after 'as'".to_string()),
        };
        self.next_token();
        let mut lazy = false;
        if self.current_token == Token::Lazy {
            self.next_token();
            if self.current_token != Token::Equals {
                return Err("Expected '=' after 'lazy'".to_string());
            }
            self.next_token();
            lazy = match &self.current_token {
                Token::True => true,
                Token::False => false,
                _ => return Err("Expected 'True' or 'False' after 'lazy ='".to_string()),
            };
            self.next_token();
        }
        Ok(AstNode::LoadDataset { path, name, lazy })
    }

    fn parse_create(&mut self) -> Result<AstNode, String> {
        self.next_token();
        if self.current_token != Token::Model {
            return Err("Expected 'model' after 'create'".to_string());
        }
        self.next_token();
        let model_type = match &self.current_token {
            Token::Identifier(id) => id.clone(),
            _ => return Err("Expected model type after 'model'".to_string()),
        };
        self.next_token();
        Ok(AstNode::CreateModel { model_type })
    }

    fn parse_train(&mut self) -> Result<AstNode, String> {
        self.next_token();
        let mut epochs = 0;
        let mut batch_size = 0;
        let mut learning_rate = 0.0;
        while self.current_token != Token::EOF {
            match &self.current_token {
                Token::Identifier(id) if id == "epochs" => {
                    self.next_token();
                    if self.current_token != Token::Equals {
                        return Err("Expected '=' after 'epochs'".to_string());
                    }
                    self.next_token();
                    epochs = match &self.current_token {
                        Token::Number(n) => *n as i32,
                        _ => return Err("Expected number after 'epochs ='".to_string()),
                    };
                }
                Token::Identifier(id) if id == "batch_size" => {
                    self.next_token();
                    if self.current_token != Token::Equals {
                        return Err("Expected '=' after 'batch_size'".to_string());
                    }
                    self.next_token();
                    batch_size = match &self.current_token {
                        Token::Number(n) => *n as i32,
                        _ => return Err("Expected number after 'batch_size ='".to_string()),
                    };
                }
                Token::Identifier(id) if id == "learning_rate" => {
                    self.next_token();
                    if self.current_token != Token::Equals {
                        return Err("Expected '=' after 'learning_rate'".to_string());
                    }
                    self.next_token();
                    learning_rate = match &self.current_token {
                        Token::Number(n) => *n,
                        _ => return Err("Expected number after 'learning_rate ='".to_string()),
                    };
                }
                _ => break,
            }
            self.next_token();
        }
        Ok(AstNode::TrainModel { epochs, batch_size, learning_rate })
    }

    fn parse_evaluate(&mut self) -> Result<AstNode, String> {
        self.next_token();
        if self.current_token != Token::Identifier("metrics".to_string()) {
            return Err("Expected 'metrics' after 'evaluate'".to_string());
        }
        self.next_token();
        if self.current_token != Token::Equals {
            return Err("Expected '=' after 'metrics'".to_string());
        }
        self.next_token();
        let metrics_str = match &self.current_token {
            Token::String(s) => s.clone(),
            _ => return Err("Expected string after 'metrics ='".to_string()),
        };
        let metrics = metrics_str.split(',').map(|s| s.trim().to_string()).collect();
        self.next_token();
        Ok(AstNode::EvaluateModel { metrics })
    }

    fn parse_save(&mut self) -> Result<AstNode, String> {
        self.next_token();
        if self.current_token != Token::Model {
            return Err("Expected 'model' after 'save'".to_string());
        }
        self.next_token();
        let path = match &self.current_token {
            Token::String(s) => s.clone(),
            _ => return Err("Expected string path after 'model'".to_string()),
        };
        self.next_token();
        Ok(AstNode::SaveModel { path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::lexer::Lexer;

    #[test]
    fn test_parse_load() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        assert_eq!(ast, AstNode::LoadDataset {
            path: "iris.csv".to_string(),
            name: "iris".to_string(),
            lazy: true,
        });
    }
}