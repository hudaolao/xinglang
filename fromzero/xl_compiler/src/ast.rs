// 星算语专用编译器 · AST 节点
#[derive(Debug, Clone)]
pub struct Program(pub Vec<Stmt>);

#[derive(Debug, Clone)]
pub enum Stmt {
    Pos(String),                                     // ▸ x
    Trans { var: String, op: String, arg: u32 },     // ↦ y = matmul(x, 64)
    Flow { op: String, arg: u32 },                   // ⟶ flow(z, 32)
    OpCall { var: String, op: String, arg: u32 },    // z = layernorm(y)
}
