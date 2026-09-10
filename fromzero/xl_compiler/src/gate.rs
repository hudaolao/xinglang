// 星算语专用编译器 · 编译期门禁(内嵌)
// 避百家之短:把 Rust 校验器硬规则 + 资源预算分析内嵌进编译器本体
// 开发者违规(▸ 顺序 / 预算超支)在编译期就被拦,拿不到字节码产物
use crate::ast::{Program, Stmt};
use crate::ir;

#[derive(Debug, Clone, Copy)]
pub struct Budget { pub mem: u64, pub flops: u64, pub bw: u64 }

// 默认编译期预算(v0.1 验收:1M 前向峰值,可配置,未来进 cost.toml)
pub const BUDGET: Budget = Budget { mem: 100 * 1024, flops: 2_200_000, bw: 196_608 };

// 每算子代价(与 cost.toml v0.1 一致,单一数据源由 _cost_diff.py 保证)
pub fn cost(op: u8) -> (u64, u64, u64) {
    match op {
        1  => (65536, 2_000_000, 131072), // matmul
        2  => (4096, 4096, 8192),         // add
        3  => (8192, 24576, 8192),        // layernorm
        4  => (8192, 32768, 8192),        // softmax
        5  => (4096, 8192, 4096),         // cross_entropy
        6  => (1024, 1024, 1024),         // qreg
        7  => (0, 1024, 0),               // H
        11 => (16384, 49152, 16384),      // adamw
        _  => (0, 0, 0),
    }
}

pub fn check(prog: &Program) -> Result<Budget, String> {
    let mut seen_pos = false;
    let mut b = Budget { mem: 0, flops: 0, bw: 0 };
    for st in &prog.0 {
        // 1. ▸ 先于 ↦(符号定位先于张量变换)
        match st {
            Stmt::Pos(_) => seen_pos = true,
            Stmt::Trans { .. } if !seen_pos => return Err("门禁:▸ 必须先于 ↦(符号定位先于张量变换)".into()),
            _ => {}
        }
        // 2. 资源预算累加
        let op = match st {
            Stmt::Trans { op, .. } | Stmt::Flow { op, .. } | Stmt::OpCall { op, .. } => ir::opcode(op),
            // v0.3 标量定义/量纲声明不计资源预算(无算子)
            Stmt::Pos(_) | Stmt::Define { .. } | Stmt::DimDef { .. } => None,
        };
        if let Some(o) = op {
            let (m, f, bw) = cost(o);
            b.mem += m; b.flops += f; b.bw += bw;
        }
    }
    // 3. 预算超支拦截
    if b.mem > BUDGET.mem { return Err(format!("门禁:内存预算超支 {}B > {}B", b.mem, BUDGET.mem)); }
    if b.flops > BUDGET.flops { return Err(format!("门禁:FLOPs 预算超支 {} > {}", b.flops, BUDGET.flops)); }
    if b.bw > BUDGET.bw { return Err(format!("门禁:带宽预算超支 {}B > {}B", b.bw, BUDGET.bw)); }
    Ok(b)
}
