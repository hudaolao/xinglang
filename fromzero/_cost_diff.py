# -*- coding: utf-8 -*-
# 星算语 fromzero · P0-6 预算 diff: cost.toml vs 校验器内联 cost 一致性检查
# 规则:任一算子 mem/flops/bw 误差 >5% → 不一致 → 编译失败
import sys

INLINE = {
 'matmul':(65536,2000000,131072),'add':(4096,4096,8192),'layernorm':(8192,24576,8192),
 'softmax':(8192,32768,8192),'cross_entropy':(4096,8192,4096),'qreg':(1024,1024,1024),
 'H':(0,1024,0),'pos':(0,0,0),'trans':(8192,16384,8192),'flow':(0,0,8192),'adamw':(16384,49152,16384),
}

tbl = {}
cur = None
for line in open(r'D:\StarSnow_Home\languages\fromzero\cost.toml', encoding='utf-8'):
    line = line.strip()
    if line.startswith('[ops.'):
        cur = line[5:line.index(']')].strip(); tbl[cur] = {}
    elif cur and '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1); tbl[cur][k.strip()] = int(v.strip())

ok = True
print('== cost.toml vs Rust 校验器内联 diff ==')
for name, (m, f, b) in INLINE.items():
    t = tbl.get(name)
    if not t:
        print(f'  [MISS] cost.toml 缺算子 {name}'); ok = False; continue
    for key, val, label in [('mem', m, '内存'), ('flops', f, 'FLOPs'), ('bw', b, '带宽')]:
        tv = t.get(key, 0)
        err = abs(tv - val) / max(val, 1) * 100
        mark = 'OK' if err <= 5 else 'DIFF'
        if err > 5: ok = False
        print(f'  {name}.{key}: toml={tv} inline={val} 误差={err:.2f}% [{mark}]')

print('== 结论:', '一致(全部≤5%)' if ok else '不一致(>5%) → 编译失败', '==')
sys.exit(0 if ok else 1)
