#[derive(Debug, PartialEq)]
pub enum Token {
    Load,
    Dataset,
    As,
    Lazy,
    Equals,
    True,
    False,
    Create,
    Model,
    Train,
    Evaluate,
    Save,
    Execute,
    Plugin,
    Epochs,
    BatchSize,
    LearningRate,
    Identifier(String),
    StringLiteral(String),
    Number(f64),
    Float(f64),
    Eof,
}

pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
    current_char: Option<char>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input,
            pos: 0,
            current_char: input.chars().next(),
        }
    }

    fn advance(&mut self) {
        self.pos += 1;
        self.current_char = self.input[self.pos..].chars().next();
    }

    fn skip_whitespace(&mut self) {
        while self.current_char.map_or(false, |c| c.is_whitespace()) {
            self.advance();
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        while let Some(c) = self.current_char {
            self.skip_whitespace();
            match c {
                'l' if self.input[self.pos..].starts_with("load") => {
                    self.pos += 4;
                    tokens.push(Token::Load);
                }
                'd' if self.input[self.pos..].starts_with("dataset") => {
                    self.pos += 7;
                    tokens.push(Token::Dataset);
                }
                'a' if self.input[self.pos..].starts_with("as") => {
                    self.pos += 2;
                    tokens.push(Token::As);
                }
                'l' if self.input[self.pos..].starts_with("lazy") => {
                    self.pos += 4;
                    tokens.push(Token::Lazy);
                }
                '=' => {
                    self.advance();
                    tokens.push(Token::Equals);
                }
                't' if self.input[self.pos..].starts_with("true") => {
                    self.pos += 4;
                    tokens.push(Token::True);
                }
                'f' if self.input[self.pos..].starts_with("false") => {
                    self.pos += 5;
                    tokens.push(Token::False);
                }
                'c' if self.input[self.pos..].starts_with("create") => {
                    self.pos += 6;
                    tokens.push(Token::Create);
                }
                'm' if self.input[self.pos..].starts_with("model") => {
                    self.pos += 5;
                    tokens.push(Token::Model);
                }
                't' if self.input[self.pos..].starts_with("train") => {
                    self.pos += 5;
                    tokens.push(Token::Train);
                }
                'e' if self.input[self.pos..].starts_with("evaluate") => {
                    self.pos += 9;
                    tokens.push(Token::Evaluate);
                }
                's' if self.input[self.pos..].starts_with("save") => {
                    self.pos += 4;
                    tokens.push(Token::Save);
                }
                'e' if self.input[self.pos..].starts_with("execute") => {
                    self.pos += 7;
                    tokens.push(Token::Execute);
                }
                'p' if self.input[self.pos..].starts_with("plugin") => {
                    self.pos += 6;
                    tokens.push(Token::Plugin);
                }
                'e' if self.input[self.pos..].starts_with("epochs") => {
                    self.pos += 6;
                    tokens.push(Token::Epochs);
                }
                'b' if self.input[self.pos..].starts_with("batch_size") => {
                    self.pos += 10;
                    tokens.push(Token::BatchSize);
                }
                'l' if self.input[self.pos..].starts_with("learning_rate") => {
                    self.pos += 13;
                    tokens.push(Token::LearningRate);
                }
                '"' => {
                    self.advance();
                    let mut literal = String::new();
                    while self.current_char.map_or(false, |c| c != '"') {
                        literal.push(c);
                        self.advance();
                    }
                    self.advance(); // Skip closing quote
                    tokens.push(Token::StringLiteral(literal));
                }
                c if c.is_alphabetic() => {
                    let mut ident = String::new();
                    while self.current_char.map_or(false, |c| c.is_alphanumeric() || c == '_') {
                        ident.push(c);
                        self.advance();
                    }
                    tokens.push(Token::Identifier(ident));
                }
                c if c.is_digit(10) || c == '-' || c == '.' => {
                    let mut num = String::new();
                    while self.current_char.map_or(false, |c| c.is_digit(10) || c == '.' || c == '-') {
                        num.push(c);
                        self.advance();
                    }
                    if num.contains('.') {
                        if let Ok(f) = num.parse::<f64>() {
                            tokens.push(Token::Float(f));
                        }
                    } else if let Ok(n) = num.parse::<f64>() {
                        tokens.push(Token::Number(n));
                    }
                }
                _ => self.advance(),
            }
        }
        tokens.push(Token::Eof);
        Ok(tokens)
    }
}