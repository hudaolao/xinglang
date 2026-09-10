// 星算语 fromzero · P0-4 C#前端(多前端之一)
// 职责:把星算语源码(迷你DSL)编译成 XL01 定长字节码,交给 Rust 校验器门禁。
// 对齐:v0.1 语义规范 §4 字节码(1字节操作码 + 3字节定长操作数)。
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;

byte OpCode(string n) => n switch
{
    "matmul" => 1, "add" => 2, "layernorm" => 3, "softmax" => 4,
    "cross_entropy" => 5, "qreg" => 6, "H" => 7, "adamw" => 11,
    _ => throw new Exception($"未知算子:{n}")
};

string OpName(byte op) => op switch
{
    1 => "matmul", 2 => "add", 3 => "layernorm", 4 => "softmax",
    5 => "cross_entropy", 6 => "qreg", 7 => "H", 8 => "▸", 9 => "↦", 10 => "⟶", 11 => "adamw", _ => "?"
};

// 解析一行 → 指令列表(三符可能展开)
List<(byte op, uint operand)> ParseLine(string line)
{
    var r = new List<(byte, uint)>();
    line = line.Trim();
    if (line.Length == 0 || line.StartsWith("//")) return r;
    byte prefix = 0;
    if (line.StartsWith("▸")) { prefix = 8; line = line.Substring(1).Trim(); }
    else if (line.StartsWith("↦")) { prefix = 9; line = line.Substring(1).Trim(); }
    else if (line.StartsWith("⟶")) { prefix = 10; line = line.Substring(1).Trim(); }

    uint operand = 0;
    string opName = "";
    bool hadParen = false;
    int eq = line.IndexOf('=');
    if (eq >= 0) line = line.Substring(eq + 1).Trim();
    int lp = line.IndexOf('(');
    if (lp >= 0)
    {
        hadParen = true;
        opName = line.Substring(0, lp).Trim();
        int rp = line.IndexOf(')', lp);
        string argStr = rp >= 0 ? line.Substring(lp + 1, rp - lp - 1) : "";
        foreach (var part in argStr.Split(','))
            if (double.TryParse(part.Trim(), out double d)) operand = (uint)(d > 0 ? d : 0);
    }
    else opName = line.Trim();

    if (prefix == 8) r.Add((8, 0));                 // ▸ noop
    else if (prefix == 9) r.Add((9, 0));            // ↦ 精度标记
    else if (prefix == 10) { r.Add((10, operand)); return r; } // ⟶ 独立块级流

    // 只有算子调用(含括号)才输出算子;三符后的裸变量锚定不输出算子
    if (opName.Length > 0 && hadParen) r.Add((OpCode(opName), operand));
    return r;
}

string Sample() =>
    "// 星算语前端样例\n" +
    "▸ x\n" +
    "↦ y = matmul(x, 64)\n" +
    "z = layernorm(y)\n" +
    "q = qreg(42)\n" +
    "⟶ flow(z, 32)\n" +
    "w = adamw(1)\n";

string src = args.Length >= 1 ? File.ReadAllText(args[0], Encoding.UTF8) : Sample();
var insts = new List<(byte, uint)>();
foreach (var raw in src.Split('\n')) insts.AddRange(ParseLine(raw));

string outDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "out");
Directory.CreateDirectory(outDir);
string outPath = Path.Combine(outDir, "sample.xlbin");
using (var w = new BinaryWriter(File.Create(outPath)))
{
    w.Write(Encoding.ASCII.GetBytes("XL01"));
    w.Write((byte)1); w.Write((byte)1); w.Write((byte)1); // 三符版本
    foreach (var (op, operand) in insts)
    {
        w.Write(op);
        w.Write((byte)(operand & 0xFF));
        w.Write((byte)((operand >> 8) & 0xFF));
        w.Write((byte)((operand >> 16) & 0xFF));
    }
}
Console.WriteLine($"C#前端: {insts.Count} 条指令 → {outPath}");
foreach (var (op, operand) in insts)
    Console.WriteLine($"   {OpName(op)}(op {op}) operand={operand}");
