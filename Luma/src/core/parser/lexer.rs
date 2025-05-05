#[derive(Debug, PartialEq)]
pub enum Token {
    Load,
    Dataset,
    As,
    Lazy,
    Create,
    Model,
    Train,
    Evaluate,
    Save,
    Identifier(String),
    String(String),
    Number(f64),
    Equals,
    True,
    False,
    Comma,
    EOF,
}

pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    ch: Option<char>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input,
            position: 0,
            ch: None,
        };
        lexer.read_char();
        lexer
    }

    fn read_char(&mut self) {
        self.ch = self.input[self.position..].chars().next();
        self.position += self.ch.map_or(0, |_| 1);
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn skip_whitespace(&mut self) {
        while self.ch.map_or(false, |c| c.is_whitespace()) {
            self.read_char();
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut ident = String::new();
        while self.ch.map_or(false, |c| c.is_alphabetic() || c == '_') {
            ident.push(self.ch.unwrap());
            self.read_char();
        }
        ident
    }

    fn read_string(&mut self) -> String {
        let mut s = String::new();
        self.read_char(); // Skip opening quote
        while self.ch.map_or(false, |c| c != '"') {
            if self.ch.is_none() {
                break; // Handle unclosed string
            }
            s.push(self.ch.unwrap());
            self.read_char();
        }
        self.read_char(); // Skip closing quote
        s
    }

    fn read_number(&mut self) -> f64 {
        let mut num = String::new();
        while self.ch.map_or(false, |c| c.is_digit(10) || c == '.') {
            num.push(self.ch.unwrap());
            self.read_char();
        }
        num.parse::<f64>().unwrap_or(0.0)
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        match self.ch {
            Some('=') => {
                self.read_char();
                if self.ch == Some('=') {
                    self.read_char();
                    Token::Equals
                } else {
                    Token::Equals
                }
            }
            Some(',') => {
                self.read_char();
                Token::Comma
            }
            Some('"') => {
                let s = self.read_string();
                Token::String(s)
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "load" => Token::Load,
                    "dataset" => Token::Dataset,
                    "as" => Token::As,
                    "lazy" => Token::Lazy,
                    "create" => Token::Create,
                    "model" => Token::Model,
                    "train" => Token::Train,
                    "evaluate" => Token::Evaluate,
                    "save" => Token::Save,
                    "True" => Token::True,
                    "False" => Token::False,
                    _ => Token::Identifier(ident),
                }
            }
            Some(c) if c.is_digit(10) => {
                let num = self.read_number();
                Token::Number(num)
            }
            None => Token::EOF,
            _ => {
                self.read_char();
                Token::EOF
            }
        }
    }
}

// Unit tests for the lexer
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer() {
        let input = r#"load dataset "iris.csv" as iris lazy=True"#;
        let mut lexer = Lexer::new(input);
        let expected = vec![
            Token::Load,
            Token::Dataset,
            Token::String("iris.csv".to_string()),
            Token::As,
            Token::Identifier("iris".to_string()),
            Token::Lazy,
            Token::Equals,
            Token::True,
            Token::EOF,
        ];
        for expected_token in expected {
            let token = lexer.next_token();
            assert_eq!(token, expected_token);
        }
    }
}