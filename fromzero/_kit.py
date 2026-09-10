# -*- coding: utf-8 -*-
# 星算语 fromzero · 多前端共享内核
# 铁则: 所有前端只写语法→IR 逻辑, 字节码发射统一走标准管线
import re, os, hashlib

OPS = {"matmul":1,"add":2,"layernorm":3,"softmax":4,"cross_entropy":5,"qreg":6,"H":7,"adamw":11}

# 各语言注释前缀, 解析时跳过
COMMENT_PREFIXES = ('#', '--', '%%', '//', 'NB.', '⍝', ';;', '%', ';', '(*', '--')

def parse_lines(src):
    """统一行解析: 每行 [op arg...] → 指令(三符展开 + 算子)"""
    insts = []
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIXES):
            continue
        line = line.rstrip(';.()')          # 去掉语言尾符 (Erlang . ; Q# ; 等)
        toks = re.findall(r'[A-Za-z_]+|\d+', line)
        if not toks:
            continue
        head = toks[0].lower()
        operand = next((int(t) for t in toks[1:] if t.isdigit()), 0)
        if head == 'loc':            # ▸ noop
            insts.append((8, 0))
        elif head == 'transform':    # ↦ 精度标记
            insts.append((9, 0))
        elif head == 'flow':         # ⟶ 独立块级流
            insts.append((10, operand))
        elif head in OPS:
            insts.append((OPS[head], operand))
    return insts

def emit_xl01(insts, out_path):
    """发射 XL01 定长字节码: 4魔数 + 3符版本 + 每条(1 opcode + 3 operand LE)"""
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([1, 1, 1]))
        for op, operand in insts:
            w.write(bytes([op]))
            w.write(bytes([operand & 0xFF, (operand >> 8) & 0xFF, (operand >> 16) & 0xFF]))

def run(name, src, out_path):
    insts = parse_lines(src)
    emit_xl01(insts, out_path)
    h = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()[:16]
    print(f'{name}前端: {len(insts)} 条指令 → {out_path}')
    for op, operand in insts:
        print(f'   op={op} operand={operand}')
    print(f'   字节码哈希: {h}')
    return insts
