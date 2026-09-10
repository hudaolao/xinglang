// 星算语专用编译器 · 语法分析器 (parser)
use crate::ast::{Program, Stmt};
use crate::lexer::Tok;

pub struct Parser { toks: Vec<Tok>, pos: usize }
impl Parser {
    pub fn new(toks: Vec<Tok>) -> Self { Self { toks, pos: 0 } }
    fn peek(&self) -> &Tok { &self.toks[self.pos] }
    fn next(&mut self) -> Tok { let t = self.toks[self.pos].clone(); if self.pos + 1 < self.toks.len() { self.pos += 1; } t }

    pub fn parse(mut self) -> Program {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            match self.next() {
                Tok::Tri('▸') => {
                    if let Tok::Ident(v) = self.next() { stmts.push(Stmt::Pos(v)); }
                }
                Tok::Tri('↦') => {
                    let var = if let Tok::Ident(v) = self.next() { v } else { String::new() };
                    self.next(); // =
                    let (op, arg) = self.op_call();
                    stmts.push(Stmt::Trans { var, op, arg });
                }
                Tok::Tri('⟶') => {
                    let (op, arg) = self.op_call();
                    stmts.push(Stmt::Flow { op, arg });
                }
                Tok::Ident(var) => {
                    if matches!(self.peek(), Tok::Assign) {
                        self.next();
                        let (op, arg) = self.op_call();
                        stmts.push(Stmt::OpCall { var, op, arg });
                    }
                }
                _ => {}
            }
        }
        Program(stmts)
    }

    fn op_call(&mut self) -> (String, u32) {
        let op = if let Tok::Ident(o) = self.next() { o } else { String::new() };
        let mut arg = 0u32;
        if matches!(self.peek(), Tok::LParen) {
            self.next();
            loop {
                match self.next() {
                    Tok::Num(n) => arg = n,
                    Tok::Comma | Tok::Ident(_) => {}
                    _ => break,
                }
            }
        }
        (op, arg)
    }
}
