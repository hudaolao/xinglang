// 星算语专用编译器 · AST 节点
// v0.1: 四条训练图语句 (Pos/Trans/Flow/OpCall)
// v0.3: 新增表达式 Expr + 标量定义 Define / 量纲声明 DimDef (中端量纲类型系统的载体)
#[derive(Debug, Clone)]
pub struct Program(pub Vec<Stmt>);

// 表达式: v0.3 量纲传播所需的最小算术语法树 (镜像 C# ParseExpr 结构)
#[derive(Debug, Clone)]
pub enum Expr {
    Num(u32),                                   // 数字字面量 → 无量纲
    Var(String),                                // 位置引用 → 继承声明量纲
    Bin(Box<Expr>, char, Box<Expr>),            // + - * / ^
    Neg(Box<Expr>),                             // 一元负号
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Pos(String),                                     // ▸ x  (符号定位)
    Trans { var: String, op: String, arg: u32 },     // ↦ y = matmul(x, 64)
    Flow { op: String, arg: u32 },                   // ⟶ flow(z, 32)
    OpCall { var: String, op: String, arg: u32 },    // z = layernorm(y)
    Define { var: String, expr: Expr, line: usize },                  // ▸ x = expr            (量纲由表达式推断)
    DimDef { var: String, expr: Expr, dim: String, line: usize },     // ▸ x = expr ◆GB        (显式量纲声明)
}
