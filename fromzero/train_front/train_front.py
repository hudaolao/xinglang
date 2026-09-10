# -*- coding: utf-8 -*-
# 星算语 · 训练前端 → 训练级 XL01 字节码
# P3② = P2收口: 把因果线性注意力训练表达成统一字节码算子流
# 对齐: spec/xinglang_v02_training_ir.md (算子扩展 12-16 + 训练IR结构)
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 训练 IR 算子流 (每 step): (opcode, operand, 注释)
# opcode: 1=matmul 5=cross_entropy 6=qreg 8=▸ 9=↦ 10=⟶ 11=adamw 12=rms_norm 14=linear_attn_fwd 15=linear_attn_bwd 16=clip_grad 17=embed
TRAIN_IR = [
    (6, 42,        'qreg(42)  确定性权重种子'),
    (17, 32,       'X = embed(E, S)  输入嵌入(查E表)'),
    (8, 0,         '▸ params  符号锚定参数'),
    (9, 0,         '↦ embed   神经层入口'),
    (1, 32,        'Q = matmul(Wq, X)  Q投影'),
    (1, 32,        'K = matmul(Wk, X)  K投影'),
    (1, 32,        'V = matmul(Wv, X)  V投影'),
    (12, 32,       "Q' = rms_norm(Q)   RMS归一化"),
    (12, 32,       "K' = rms_norm(K)   RMS归一化"),
    (10, 0,        '⟶ 块级流入口(禁全量驻留)'),
    (14, 32,       'y = linear_attn_fwd(Q\',K\',V)  state/z增量+den'),
    (1, 32,        'logits = matmul(Wo, y) 输出投影'),
    (5, 32,        'loss = cross_entropy(logits, S)'),
    (10, 0,        '⟶ 反向块级流'),
    (15, 32,       'g = linear_attn_bwd(...)'),
    (16, 32,       'g = clip_grad(g)  梯度裁剪'),
    (11, 1,        'w = adamw(lr)  参数更新'),
]

def emit_xl01(insts, out_path):
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([2, 0, 0]))   # v0.2 三符版本(训练级)
        for op, operand, _ in insts:
            w.write(bytes([op]))
            w.write(bytes([operand & 0xFF, (operand >> 8) & 0xFF, (operand >> 16) & 0xFF]))

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'train_step.xlbin')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emit_xl01(TRAIN_IR, out)
    raw = open(out, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()[:16]
    print(f'训练前端: {len(TRAIN_IR)} 条指令 → {out}')
    for op, operand, note in TRAIN_IR:
        print(f'   op={op:>2} operand={operand:<3} {note}')
    print(f'   字节码总长: {len(raw)} 字节')
    print(f'   训练IR哈希: {h}')

if __name__ == '__main__':
    main()
