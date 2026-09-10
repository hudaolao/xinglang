// 星算语专用编译器 · 入口
// 用法: xl_compiler [源码文件]   -> 生成 out.xlbin (XL01 字节码),供 Rust 校验器门禁
mod lexer; mod ast; mod parser; mod ir; mod gate; mod type_system;
use std::io::Write;
use std::path::Path;

const SAMPLE: &str = "// 星算语编译器自测样例\n▸ x\n↦ y = matmul(x, 64)\nz = layernorm(y)\nq = qreg(42)\n⟶ flow(z, 32)\nw = adamw(1)\n";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = if args.len() >= 2 { std::fs::read_to_string(&args[1]).unwrap_or_default() } else { SAMPLE.to_string() };

    let toks = lexer::tokenize(&src);
    let prog = parser::Parser::new(toks).parse();
    println!("== AST ==");
    for st in &prog.0 { println!("   {:?}", st); }

    // 编译期门禁:▸ 顺序 + 资源预算(编译失败则拿不到字节码产物)
    match gate::check(&prog) {
        Ok(b) => println!("== 门禁通过: 预算 mem={}B flops={} bw={}B ==", b.mem, b.flops, b.bw),
        Err(e) => { eprintln!("{}", e); std::process::exit(1); }
    }

    // v0.3 量纲类型系统: 编译期量纲传播 + 不匹配拦截(带行号;失败则拿不到字节码产物)
    match type_system::check(&prog) {
        Ok(()) => println!("== 量纲门禁通过: 表达式量纲自洽 =="),
        Err(e) => { eprintln!("{}", e); std::process::exit(1); }
    }

    let irc = ir::emit(&prog);
    println!("== IR: {} 条 ==", irc.len());
    for (op, arg) in &irc { println!("   {} op={} arg={}", ir::opname(*op), op, arg); }

    let out = Path::new("out.xlbin");
    let mut f = std::fs::File::create(out).expect("创建 out.xlbin 失败");
    f.write_all(b"XL01").expect("写头失败");
    f.write_all(&[1, 1, 1]).expect("写版本失败");
    for (op, arg) in &irc {
        f.write_all(&[*op, (arg & 0xFF) as u8, ((arg >> 8) & 0xFF) as u8, ((arg >> 16) & 0xFF) as u8]).expect("写指令失败");
    }
    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!("== 写出 {} ({}B) ==", out.display(), size);
}
