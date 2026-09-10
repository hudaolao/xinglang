# -*- coding: utf-8 -*-
# 星算语 · Lisp 前端(纯函数/惰性源头) → XL01 字节码
# P3 多源头汇合: 同一逻辑程序, Lisp 风格源码 → 与 cs_front/APL 前端字节码哈希一致
import re, sys, os, hashlib

OPS = {"matmul":1,"add":2,"layernorm":3,"softmax":4,"cross_entropy":5,"qreg":6,"H":7,"adamw":11}

# Lisp 风格源码: 每条指令一个 S 表达式 (op arg...)
LISP_SRC = r"""
;; 星算语 Lisp 前端 · 同一逻辑程序(对照 C# 前端 sample)
(loc x)              ; ▸ x
(transform)          ; ↦ 精度标记
(matmul x 64)        ; y = matmul(x, 64)
(layernorm y)        ; z = layernorm(y)
(qreg 42)            ; q = qreg(42)
(flow z 32)          ; ⟶ flow(z, 32)
(adamw 1)            ; w = adamw(1)
"""

def parse_lisp(src):
    insts = []
    for m in re.finditer(r'\(([^()]*)\)', src):
        parts = m.group(1).split()
        if not parts:
            continue
        head = parts[0].lower()
        if head == 'loc':            # ▸ noop
            insts.append((8, 0))
        elif head == 'transform':    # ↦ 精度标记
            insts.append((9, 0))
        elif head == 'flow':         # ⟶ 独立块级流
            operand = next((int(a) for a in parts[1:] if a.isdigit()), 0)
            insts.append((10, operand))
        elif head in OPS:
            operand = next((int(a) for a in parts[1:] if a.isdigit()), 0)
            insts.append((OPS[head], operand))
    return insts

def emit_xl01(insts, out_path):
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([1, 1, 1]))  # 三符版本
        for op, operand in insts:
            w.write(bytes([op]))
            w.write(bytes([operand & 0xFF, (operand >> 8) & 0xFF, (operand >> 16) & 0xFF]))

def main():
    insts = parse_lisp(LISP_SRC)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'lisp.xlbin')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emit_xl01(insts, out)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]
    print(f'Lisp前端: {len(insts)} 条指令 → {out}')
    for op, operand in insts:
        print(f'   op={op} operand={operand}')
    print(f'   字节码哈希: {h}')
    return insts

if __name__ == '__main__':
    main()
