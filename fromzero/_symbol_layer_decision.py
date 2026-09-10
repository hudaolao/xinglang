# -*- coding: utf-8 -*-
# 星算语 fromzero · 符号层命运决策
# 解析 train_step.xlbin, 构建计算图依赖, 统计各算子 FLOPs 占比
# 判定: ▸ 符号层是否 >20% FLOPs 可削 → 稀疏化优先 or 推迟
import struct, os

BASE = r'E:\StarSnow_Home\languages\fromzero'
IR = os.path.join(BASE, 'train_front', 'out', 'train_step.xlbin')
V, D, L = 32, 32, 16

NAMES = {1:'matmul',2:'add',3:'layernorm',4:'softmax',5:'cross_entropy',6:'qreg',
         7:'H',8:'▸符号',9:'↦神经',10:'⟶量子流',11:'adamw',12:'rms_norm',13:'elu',
         14:'linear_attn_fwd',15:'linear_attn_bwd',16:'clip_grad',17:'embed'}

# 读指令
insts = []
data = open(IR, 'rb').read()
assert data[:4] == b'XL01', '非XL01'
i = 7
while i < len(data):
    op = data[i]; i += 1
    opd = data[i] | (data[i+1] << 8) | (data[i+2] << 16); i += 3
    insts.append((op, opd))

print('== 星算语 符号层命运决策 ==')
print(f'IR: {IR}  指令 {len(insts)} 条')

# FLOPs 估算 (每 step, 按算子语义)
# 依赖: 每个算子读/写哪些张量(计算图)
def flops(op, cnt):
    if op == 1:   # matmul: Q/K/V 投影 + logits, 分位置
        return 3 * (2*L*D*D) + (2*L*D*V)   # 3投影 + 1输出
    if op == 12:  # rms_norm: 均方根 + 除
        return 2 * (2*L*D + L*D)
    if op == 14:  # attn_fwd: state累积(L*D^2*2) + NUM(L*D^2*2) + DEN
        return L*D*D*2 + L*D*D*2 + L*D
    if op == 5:   # cross_entropy: softmax + log
        return L*V*2
    if op == 15:  # attn_bwd: 反向 state/梯度
        return 4 * L*D*D
    if op == 11:  # adamw: 逐参数 ~5 ops * 5 组
        return 5 * (V*D + 3*D*D + V*D) * 5
    if op == 16:  # clip_grad: 二范数 + 缩放
        return 3 * (V*D + 3*D*D + V*D)
    if op == 17:  # embed: gather
        return L*D
    if op == 6:   # qreg: 初始化, 每 step 不计(仅首次)
        return 0
    # ▸/↦/⟶ (8/9/10) 及 noop: 0 FLOPs
    return 0

# 依赖: 算子读写张量 (计算图边)
READS = {1:['X','Wq','Wk','Wv','Y','Wo'], 12:['Q','K'], 14:['K','VV','Q'],
         5:['LG'], 15:['LG','Wo','Y','DEN','NUM','STATE','Z','Q','K','VV','X','Wq','Wk','Wv','RQ','RK'],
         16:['dE','dWq','dWk','dWv','dWo'], 11:['dE','dWq','dWk','dWv','dWo'],
         17:['E','S'], 6:[]}
WRITES = {1:['Q','K','VV','LG'], 12:['Q','K'], 14:['STATE','Z','NUM','DEN','Y'],
          5:['dLG'], 15:['dWq','dWk','dWv','dWo','dE'], 16:['dE','dWq','dWk','dWv','dWo'],
          11:['E','Wq','Wk','Wv','Wo'], 17:['X'], 6:['E','Wq','Wk','Wv','Wo']}

from collections import Counter
fmap = Counter(); reads = Counter(); writes = Counter()
for op, opd in insts:
    n = NAMES.get(op, f'op{op}')
    fmap[n] += flops(op, 1)
    for t in READS.get(op, []): reads[t] += 1
    for t in WRITES.get(op, []): writes[t] += 1

total = sum(fmap.values())
print('\n== 各算子 FLOPs 占比 (每 step) ==')
rows = sorted(fmap.items(), key=lambda kv: -kv[1])
for n, f in rows:
    pct = 100.0*f/total if total else 0
    bar = '#'*int(pct/3)
    print(f'  {n:<16} {f:>9,}  {pct:5.1f}%  {bar}')
print(f'  {"合计":<16} {total:>9,}')

# 符号层(▸/↦/⟶) 占比
sym = fmap['▸符号'] + fmap['↦神经'] + fmap['⟶量子流']
sym_pct = 100.0*sym/total if total else 0

print('\n== 计算图依赖 (张量被算子读/写次数) ==')
for t in sorted(set(reads)|set(writes)):
    print(f'  {t:<8} 读{reads[t]:>2} 写{writes[t]:>2}')

print('\n== 决策 ==')
print(f'符号层(▸/↦/⟶) FLOPs: {sym_pct:.2f}%  (>20% 才考虑削)')
if sym_pct > 20:
    print('  → 符号层 >20% FLOPs 可削 → 稀疏化优先, 削减 noop 符号层')
else:
    print('  → 符号层 ≈0%, 本身是 noop 签名冻结, 削无可削 → 保持冻结, 其价值在结构标记而非计算')
print('  真正的 FLOPs 大头: matmul + linear_attn_fwd/bwd + adamw')
print('  10M 真训可削方向: attn_bwd 的 O(L*D^2) 反向累积(拟态/chunk近似), adamw 为必要不可削')
