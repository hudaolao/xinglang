// 星算语专用编译器 · IR 生成 (XL01 定长字节码)
// 对齐 v0.1 语义规范 §4:1字节操作码 + 3字节定长操作数
use crate::ast::{Program, Stmt};

pub fn opcode(op: &str) -> Option<u8> {
    Some(match op {
        "matmul" => 1, "add" => 2, "layernorm" => 3, "softmax" => 4,
        "cross_entropy" => 5, "qreg" => 6, "H" => 7, "adamw" => 11,
        _ => return None,
    })
}
pub fn opname(op: u8) -> &'static str {
    match op { 1 => "matmul", 2 => "add", 3 => "layernorm", 4 => "softmax", 5 => "cross_entropy",
               6 => "qreg", 7 => "H", 8 => "▸", 9 => "↦", 10 => "⟶", 11 => "adamw", _ => "?" }
}

pub fn emit(prog: &Program) -> Vec<(u8, u32)> {
    let mut ir = Vec::new();
    for st in &prog.0 {
        match st {
            Stmt::Pos(_) => ir.push((8, 0)),
            Stmt::Trans { op, arg, .. } => { ir.push((9, 0)); if let Some(o) = opcode(op) { ir.push((o, *arg)); } }
            Stmt::Flow { op, arg } => { ir.push((10, *arg)); if let Some(o) = opcode(op) { ir.push((o, 0)); } }
            Stmt::OpCall { op, arg, .. } => { if let Some(o) = opcode(op) { ir.push((o, *arg)); } }
        }
    }
    ir
}
