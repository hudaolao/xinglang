// 星算语专用编译器 · 语法分析器 (parser)
// v0.1: ▸ 位置 / ↦ 变换 / ⟶ 流 / var = op(...)  四类语句
// v0.3: ▸ ident = expr [◆DIM]  定义/量纲声明 + 递归下降表达式
use crate::ast::{Program, Stmt, Expr};
use crate::lexer::{Tok, Spanned};

pub struct Parser { toks: Vec<Spanned>, pos: usize }
impl Parser {
    pub fn new(toks: Vec<Spanned>) -> Self { Self { toks, pos: 0 } }
    fn peek(&self) -> &Tok { &self.toks[self.pos].tok }
    fn line(&self) -> usize { self.toks[self.pos].line }
    fn next(&mut self) -> Tok { let t = self.toks[self.pos].tok.clone(); if self.pos + 1 < self.toks.len() { self.pos += 1; } t }

    pub fn parse(mut self) -> Program {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            let line = self.line();
            match self.next() {
                Tok::Tri('▸') => {
                    if let Tok::Ident(v) = self.next() {
                        // ▸ x  (纯位置)  vs  ▸ x = expr [◆DIM]  (定义/量纲声明)
                        if matches!(self.peek(), Tok::Assign) {
                            self.next(); // =
                            let expr = self.parse_expr();
                            if matches!(self.peek(), Tok::DimSig) {
                                self.next(); // ◆
                                let dim = if let Tok::DimName(d) = self.next() { d } else { String::new() };
                                stmts.push(Stmt::DimDef { var: v, expr, dim, line });
                            } else {
                                stmts.push(Stmt::Define { var: v, expr, line });
                            }
                        } else {
                            stmts.push(Stmt::Pos(v));
                        }
                    }
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

    // ── v0.3 递归下降表达式 (镜像 C# ParseExpr 优先级) ──
    // expr = term (('+'|'-') term)* ; term = factor (('*'|'/') factor)*
    // factor = ('-')? atom ('^' factor)? ; atom = Num | Var | '(' expr ')'
    fn parse_expr(&mut self) -> Expr {
        let mut left = self.parse_term();
        while matches!(self.peek(), Tok::Plus | Tok::Minus) {
            let op = if matches!(self.peek(), Tok::Plus) { '+' } else { '-' };
            self.next();
            let right = self.parse_term();
            left = Expr::Bin(Box::new(left), op, Box::new(right));
        }
        left
    }
    fn parse_term(&mut self) -> Expr {
        let mut left = self.parse_factor();
        while matches!(self.peek(), Tok::Star | Tok::Slash) {
            let op = if matches!(self.peek(), Tok::Star) { '*' } else { '/' };
            self.next();
            let right = self.parse_factor();
            left = Expr::Bin(Box::new(left), op, Box::new(right));
        }
        left
    }
    fn parse_factor(&mut self) -> Expr {
        if matches!(self.peek(), Tok::Minus) {
            self.next();
            return Expr::Neg(Box::new(self.parse_factor()));
        }
        let mut atom = self.parse_atom();
        if matches!(self.peek(), Tok::Caret) {
            self.next();
            let e = self.parse_factor();
            atom = Expr::Bin(Box::new(atom), '^', Box::new(e));
        }
        atom
    }
    fn parse_atom(&mut self) -> Expr {
        match self.peek().clone() {
            Tok::Num(n) => { self.next(); Expr::Num(n) }
            Tok::Ident(v) => { self.next(); Expr::Var(v) }
            Tok::LParen => {
                self.next();
                let e = self.parse_expr();
                if matches!(self.peek(), Tok::RParen) { self.next(); }
                e
            }
            _ => { self.next(); Expr::Num(0) }
        }
    }
}
