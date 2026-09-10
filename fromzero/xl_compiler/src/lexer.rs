// 星算语专用编译器 · 词法分析器 (lexer)
// v0.1: 三符 ▸↦⟶ 一等 token;算子/变量为标识符;支持 // 注释
// v0.3: 加行号(Spanned) 供量纲诊断带行号;新增 ◆ 量纲签名 + 表达式运算符 + - * / ^
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Tri(char),       // ▸ ↦ ⟶
    Ident(String),
    Num(u32),
    Assign,
    LParen,
    RParen,
    Comma,
    Plus,            // +   v0.3 表达式
    Minus,           // -
    Star,            // *
    Slash,           // /
    Caret,           // ^
    DimSig,          // ◆   v0.3 量纲声明签名
    DimName(String), // GB / MB / B / % / min ... (◆ 之后紧跟的量纲名)
    Eof,
}

// 带源位置的 token:中端语义门禁(量纲检查)报错时可回指行号
#[derive(Debug, Clone)]
pub struct Spanned {
    pub tok: Tok,
    pub line: usize,
}

pub fn tokenize(src: &str) -> Vec<Spanned> {
    let mut out = Vec::new();
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    while i < cs.len() {
        let c = cs[i];
        match c {
            ' ' | '\t' => i += 1,
            '\n' => { line += 1; i += 1; }
            '\r' => i += 1,
            // // 注释:吞到行尾(不吞 \n,交给上面的 '\n' 分支计入行号)
            '/' if i + 1 < cs.len() && cs[i + 1] == '/' => {
                while i < cs.len() && cs[i] != '\n' { i += 1; }
            }
            '▸' | '↦' | '⟶' => { out.push(Spanned { tok: Tok::Tri(c), line }); i += 1; }
            // ◆ 量纲签名:紧接着把 [字母%]+ 收为一个量纲名 token
            '◆' => {
                out.push(Spanned { tok: Tok::DimSig, line });
                i += 1;
                let mut s = String::new();
                while i < cs.len() && (cs[i].is_ascii_alphabetic() || cs[i] == '%') {
                    s.push(cs[i]); i += 1;
                }
                if !s.is_empty() { out.push(Spanned { tok: Tok::DimName(s), line }); }
            }
            '=' => { out.push(Spanned { tok: Tok::Assign, line }); i += 1; }
            '(' => { out.push(Spanned { tok: Tok::LParen, line }); i += 1; }
            ')' => { out.push(Spanned { tok: Tok::RParen, line }); i += 1; }
            ',' => { out.push(Spanned { tok: Tok::Comma, line }); i += 1; }
            '+' => { out.push(Spanned { tok: Tok::Plus, line }); i += 1; }
            '-' => { out.push(Spanned { tok: Tok::Minus, line }); i += 1; }
            '*' => { out.push(Spanned { tok: Tok::Star, line }); i += 1; }
            // 注意:独立 '/' 不会落到这里,因为 // 注释已在上面先匹配
            '/' => { out.push(Spanned { tok: Tok::Slash, line }); i += 1; }
            '^' => { out.push(Spanned { tok: Tok::Caret, line }); i += 1; }
            c if c.is_ascii_digit() => {
                let mut s = String::new();
                while i < cs.len() && cs[i].is_ascii_digit() { s.push(cs[i]); i += 1; }
                out.push(Spanned { tok: Tok::Num(s.parse().unwrap_or(0)), line });
            }
            c if c.is_ascii_alphanumeric() || c == '_' || c == '.' => {
                let mut s = String::new();
                while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_' || cs[i] == '.') { s.push(cs[i]); i += 1; }
                out.push(Spanned { tok: Tok::Ident(s), line });
            }
            _ => i += 1,
        }
    }
    out.push(Spanned { tok: Tok::Eof, line });
    out
}
