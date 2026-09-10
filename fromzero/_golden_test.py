# -*- coding: utf-8 -*-
# 星算语 fromzero · P0-7 Golden 测试骨架
# 覆盖已建立的确定性铁律:C核matmul golden / QReg种子映射 / cost一致性 / 前端→门禁链路
import json, subprocess, sys, os

BASE = r'D:\StarSnow_Home\languages\fromzero'
results = []

def check(name, cond, detail=''):
    results.append((name, cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  <- {detail}' if detail else ''))

print('== 星算语 fromzero · Golden 测试骨架 ==')

# 1. C核matmul golden(确定性并行)
gp = os.path.join(BASE, 'golden', 'matmul_bench_golden.json')
if os.path.exists(gp):
    g = json.load(open(gp, encoding='utf-8'))
    check('C核matmul golden哈希', g.get('golden_hash') == 'efd369beca5c7ed5', g.get('golden_hash'))
    check('C核matmul 逐位一致', g.get('bits_equal') is True)
else:
    check('C核matmul golden文件', False, '缺失')

# 2. QReg 种子映射 golden(seed=42)
qg = ['0x7f7aa885', '0xaf88544a', '0xdbdfefb8', '0xdc5aabb0', '0xbd67d2db', '0x7dd7d086']
qp = os.path.join(BASE, 'spec', 'qreg_seed_contract.md')
if os.path.exists(qp):
    txt = open(qp, encoding='utf-8').read()
    check('QReg seed=42 golden前6值', all(h in txt for h in qg))
else:
    check('QReg合同文件', False, '缺失')

# 3. cost.toml 与校验器内联一致性(误差≤5%)
rd = subprocess.run([sys.executable, os.path.join(BASE, '_cost_diff.py')],
                    capture_output=True, text=True, encoding='utf-8')
check('cost.toml 预算diff一致', rd.returncode == 0)

# 4. C#前端产物(前端→门禁链路可握手)
xl = os.path.join(BASE, 'cs_front', 'out', 'sample.xlbin')
check('C#前端产物 sample.xlbin', os.path.exists(xl), xl)

# 5. Rust校验器源码在岗
check('Rust校验器源码', os.path.exists(os.path.join(BASE, 'rust_gate', 'src', 'main.rs')))

total = len(results); passed = sum(1 for _, c in results if c)
print(f'\n== 结果: {passed}/{total} 通过 ==')
sys.exit(0 if passed == total else 1)
