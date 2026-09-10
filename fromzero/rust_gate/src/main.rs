// 星算语 fromzero · P0-3 Rust 校验器(静态门禁)
// 硬规则:①字节码格式合法 ②三符执行顺序(▸ 必须先于同作用域 ↦) ③资源预算累加(超预算编译失败) ④⟶ 禁全量驻留
// 架构对齐:v0.1 语义规范 §4(字节码) §5(确定性) §6(资源签名/代价语义)
use std::env;

const OP_MATMUL: u8 = 1; const OP_ADD: u8 = 2; const OP_LAYERNORM: u8 = 3;
const OP_SOFTMAX: u8 = 4; const OP_CROSS_ENTROPY: u8 = 5; const OP_QREG: u8 = 6;
const OP_H: u8 = 7; const OP_POS: u8 = 8; const OP_TRANS: u8 = 9; const OP_FLOW: u8 = 10; const OP_ADAMW: u8 = 11;

fn op_name(op: u8) -> &'static str {
    match op {
        OP_MATMUL => "matmul", OP_ADD => "add", OP_LAYERNORM => "layernorm",
        OP_SOFTMAX => "softmax", OP_CROSS_ENTROPY => "cross_entropy", OP_QREG => "qreg",
        OP_H => "H", OP_POS => "▸", OP_TRANS => "↦", OP_FLOW => "⟶", OP_ADAMW => "adamw",
        _ => "?" }
}

// 资源签名:(内存增量 B 上界, FLOPs, 带宽 B) —— 对应 cost.toml 概念,先内联,后接外部 cost.toml
fn cost(op: u8) -> Option<(u64, u64, u64)> {
    match op {
        OP_MATMUL => Some((65536, 2_000_000, 131072)),
        OP_ADD => Some((4096, 4096, 8192)),
        OP_LAYERNORM => Some((8192, 24576, 8192)),
        OP_SOFTMAX => Some((8192, 32768, 8192)),
        OP_CROSS_ENTROPY => Some((4096, 8192, 4096)),
        OP_QREG => Some((1024, 1024, 1024)),
        OP_H => Some((0, 1024, 0)),
        OP_POS => Some((0, 0, 0)),      // ▸ noop:零内存,但必须产生 trace
        OP_TRANS => Some((8192, 16384, 8192)),
        OP_FLOW => Some((0, 0, 8192)),  // ⟶ 块级流:零驻留内存
        OP_ADAMW => Some((16384, 49152, 16384)),
        _ => None }
}

#[derive(Clone, Copy, Debug)]
struct Inst { op: u8, operand: u32 }

fn parse_bytecode(data: &[u8]) -> Result<Vec<Inst>, String> {
    if data.len() < 7 { return Err("字节码太短:缺头部".into()); }
    if &data[0..4] != b"XL01" {
        return Err(format!("magic 错误:应为 XL01,实际 {}", String::from_utf8_lossy(&data[0..4])));
    }
    let mut v = Vec::new(); let mut i = 7;
    while i + 4 <= data.len() {
        v.push(Inst { op: data[i], operand: u32::from_le_bytes([data[i+1], data[i+2], data[i+3], 0]) });
        i += 4;
    }
    if i != data.len() { return Err("指令序列不是 4 字节定长(尾字节残留)".into()); }
    Ok(v)
}

fn gate(insts: &[Inst], bm: u64, bf: u64, bb: u64) -> (bool, Vec<String>, u64, u64, u64) {
    let mut ok = true; let mut msgs = Vec::new();
    let (mut m, mut f, mut b) = (0u64, 0u64, 0u64);
    let mut seen_pos = false;
    for (idx, ins) in insts.iter().enumerate() {
        let c = match cost(ins.op) { Some(c) => c, None => { ok = false; msgs.push(format!("[{idx}] 非法操作码 {}", ins.op)); continue; } };
        // ② 三符顺序:▸ 必须先于同作用域任何 ↦
        if ins.op == OP_TRANS && !seen_pos { ok = false; msgs.push(format!("[{idx}] ↦ 出现前缺少 ▸(三符顺序违反)")); }
        if ins.op == OP_POS { seen_pos = true; }
        // ④ ⟶ 禁全量驻留:3字节操作数最高位(bit23=0x800000)置位 = 声明全量驻留 → 拦截
        if ins.op == OP_FLOW && (ins.operand & 0x80_0000) != 0 { ok = false; msgs.push(format!("[{idx}] ⟶ 声明全量驻留(禁止),必须块级流")); }
        m += c.0; f += c.1; b += c.2;
    }
    if m > bm { ok = false; msgs.push(format!("内存预算超支: {m}B > {bm}B")); }
    if f > bf { ok = false; msgs.push(format!("FLOPs 预算超支: {f} > {bf}")); }
    if b > bb { ok = false; msgs.push(format!("带宽预算超支: {b}B > {bb}B")); }
    (ok, msgs, m, f, b)
}

fn run(data: &[u8]) {
    match parse_bytecode(data) {
        Err(e) => { println!("校验失败: {e}"); }
        Ok(insts) => {
            let (ok, msgs, m, f, b) = gate(&insts, 1_000_000, 10_000_000, 1_000_000);
            println!("指令数={}  资源签名: 内存={m}B FLOPs={f} 带宽={b}B", insts.len());
            for ins in &insts { println!("   {}(op {})", op_name(ins.op), ins.op); }
            if ok { println!("✔ 校验通过"); }
            else { println!("✘ 校验失败: {} 条", msgs.len()); for x in &msgs { println!("   - {x}"); } }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        let data = std::fs::read(&args[1]).expect("读字节码文件失败");
        run(&data);
    } else {
        println!("== 校验器自测 ==\n[合法样例: ▸→↦→matmul→QReg→adamw]");
        let legal = [OP_POS, OP_TRANS, OP_MATMUL, OP_QREG, OP_ADAMW];
        let mut d = vec![b'X', b'L', b'0', b'1', 1, 1, 1];
        for &o in &legal { d.push(o); d.push(0); d.push(0); d.push(0); }
        run(&d);
        println!("\n[非法样例1: ↦ 前无 ▸]");
        let illegal1 = [OP_TRANS, OP_MATMUL];
        let mut d2 = vec![b'X', b'L', b'0', b'1', 1, 1, 1];
        for &o in &illegal1 { d2.push(o); d2.push(0); d2.push(0); d2.push(0); }
        run(&d2);
        println!("\n[非法样例2: ⟶ 声明全量驻留]");
        let mut d3 = vec![b'X', b'L', b'0', b'1', 1, 1, 1];
        d3.push(OP_POS); d3.push(0); d3.push(0); d3.push(0);
        d3.push(OP_FLOW); d3.push(0); d3.push(0); d3.push(0x80); // 3字节操作数最高位=全量驻留标志
        run(&d3);
    }
}
