// 星算语专用编译器 · v0.3 量纲类型系统 (中端语义门禁)
// 镜像 C# 参考实现 v0.10 量纲语义 (Program.cs DimOf):
//   - 加减: 两侧量纲非空且不同 → 编译期报错 (1e9 换算病/量纲丢失在此被根除)
//   - 乘除: 同量纲继承; 异量纲 → 导出量纲(不追踪复合,记无量纲)
//   - 幂 ^ : 保留左量纲; 一元负号: 量纲不变; 数字字面量: 无量纲; 位置引用: 继承声明量纲
// 设计哲学呼应设计文档 §测量律: 编译器在编译期"测量"出量纲错误(静态检查),运行期才测量数值。
use crate::ast::{Program, Stmt, Expr};
use std::collections::HashMap;

// None = 无量纲; Some("GB") = 带量纲
type Dim = Option<String>;

/// 逐语句跑符号表 + 量纲传播; 任一量纲矛盾 → 带行号的编译错
pub fn check(prog: &Program) -> Result<(), String> {
    let mut env: HashMap<String, Dim> = HashMap::new();
    for st in &prog.0 {
        match st {
            // 算子训练图语句在 v0.3 不参与标量量纲检查(其张量量纲留待 v0.4)
            Stmt::Pos(_) | Stmt::Trans { .. } | Stmt::Flow { .. } | Stmt::OpCall { .. } => {}
            Stmt::Define { var, expr, line } => {
                let d = dim_of_expr(&env, expr)
                    .map_err(|e| format!("[第 {} 行] {}", line, e))?;
                env.insert(var.clone(), d);
            }
            Stmt::DimDef { var, expr, dim, line } => {
                // 先查表达式内部量纲自洽(如 a(GB) + b(MB) 在表达式内部就爆)
                dim_of_expr(&env, expr).map_err(|e| format!("[第 {} 行] {}", line, e))?;
                // 用户显式声明量纲为准,记入符号表供后续引用继承
                env.insert(var.clone(), Some(dim.clone()));
            }
        }
    }
    Ok(())
}

fn dim_of_expr(env: &HashMap<String, Dim>, e: &Expr) -> Result<Dim, String> {
    match e {
        Expr::Num(_) => Ok(None),
        Expr::Var(v) => match env.get(v) {
            Some(d) => Ok(d.clone()),
            None => Err(format!("量纲检查:引用了未声明的位置 `{}`", v)),
        },
        Expr::Neg(inner) => dim_of_expr(env, inner), // 一元负号:量纲不变
        Expr::Bin(l, op, r) => {
            let lv = dim_of_expr(env, l)?;
            let rv = dim_of_expr(env, r)?;
            match *op {
                '+' | '-' => match (&lv, &rv) {
                    (Some(a), Some(b)) if a != b => {
                        Err(format!("量纲不匹配: {} {} {} (不同量纲不可加减)", a, op, b))
                    }
                    _ => Ok(lv.clone().or(rv.clone())), // 一边无量纲 → 继承有量纲那边
                },
                '*' | '/' => match (&lv, &rv) {
                    (Some(a), Some(b)) if a != b => Ok(None), // 异量纲乘除 → 导出(不追踪复合)
                    _ => Ok(lv.clone().or(rv.clone())),        // 同量纲继承 / 一边无量纲继承
                },
                '^' => Ok(lv.clone()), // 幂保留左量纲
                _ => Ok(None),
            }
        }
    }
}
