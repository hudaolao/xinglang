# -*- coding: utf-8 -*-
# 星算语 · APL 前端(符号化数组源头) → XL01 字节码
# P3 多源头汇合: 同一逻辑程序, APL 风格源码 → 与 cs_front/Lisp 前端字节码哈希一致
import re, sys, os, hashlib

OPS = {"matmul":1,"add":2,"layernorm":3,"softmax":4,"cross_entropy":5,"qreg":6,"H":7,"adamw":11}

# APL 风格源码: 三符前缀 + ← 赋值, 同一逻辑程序
APL_SRC = r"""
⍝ 星算语 APL 前端 · 同一逻辑程序(对照 C# 前端 sample)
▸ x
↦ y ← matmul x 64
z ← layernorm y
q ← qreg 42
⟶ flow z 32
w ← adamw 1
"""

def parse_apl(src):
    insts = []
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith('⍝'):
            continue
        prefix = 0
        if line.startswith('▸'):
            prefix = 8
            line = line[1:].strip()
        elif line.startswith('↦'):
            prefix = 9
            line = line[1:].strip()
        elif line.startswith('⟶'):
            prefix = 10
            line = line[1:].strip()
        # 去掉赋值左值 (w ← ... 或 y ← ...)
        if '←' in line:
            line = line.split('←', 1)[1].strip()
        # 取算子名和操作数
        m = re.match(r'([a-zA-Z_]+)\s+(.*)', line)
        op = m.group(1).lower() if m else line
        rest = m.group(2) if m else ''
        operand = next((int(a) for a in re.findall(r'\d+', rest)), 0)
        if prefix == 8:
            insts.append((8, 0))
        elif prefix == 9:
            insts.append((9, 0))
            if op in OPS:
                insts.append((OPS[op], operand))
        elif prefix == 10:
            insts.append((10, operand))
        elif op in OPS:
            insts.append((OPS[op], operand))
    return insts

def emit_xl01(insts, out_path):
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([1, 1, 1]))
        for op, operand in insts:
            w.write(bytes([op]))
            w.write(bytes([operand & 0xFF, (operand >> 8) & 0xFF, (operand >> 16) & 0xFF]))

def main():
    insts = parse_apl(APL_SRC)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'apl.xlbin')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emit_xl01(insts, out)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]
    print(f'APL前端: {len(insts)} 条指令 → {out}')
    for op, operand in insts:
        print(f'   op={op} operand={operand}')
    print(f'   字节码哈希: {h}')
    return insts

if __name__ == '__main__':
    main()
