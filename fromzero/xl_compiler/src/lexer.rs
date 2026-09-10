// 星算语专用编译器 · 词法分析器 (lexer)
// 三符 ▸↦⟶ 是一等 token;算子/变量为标识符;支持 // 注释
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Tri(char),       // ▸ ↦ ⟶
    Ident(String),
    Num(u32),
    Assign,
    LParen,
    RParen,
    Comma,
    Eof,
}

pub fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '/' if i + 1 < cs.len() && cs[i + 1] == '/' => {
                while i < cs.len() && cs[i] != '\n' { i += 1; }
            }
            '▸' | '↦' | '⟶' => { out.push(Tok::Tri(c)); i += 1; }
            '=' => { out.push(Tok::Assign); i += 1; }
            '(' => { out.push(Tok::LParen); i += 1; }
            ')' => { out.push(Tok::RParen); i += 1; }
            ',' => { out.push(Tok::Comma); i += 1; }
            c if c.is_ascii_digit() => {
                let mut s = String::new();
                while i < cs.len() && cs[i].is_ascii_digit() { s.push(cs[i]); i += 1; }
                out.push(Tok::Num(s.parse().unwrap_or(0)));
            }
            c if c.is_ascii_alphanumeric() || c == '_' || c == '.' => {
                let mut s = String::new();
                while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_' || cs[i] == '.') { s.push(cs[i]); i += 1; }
                out.push(Tok::Ident(s));
            }
            _ => i += 1,
        }
    }
    out.push(Tok::Eof);
    out
}
