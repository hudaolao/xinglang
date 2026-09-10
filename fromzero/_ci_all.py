# -*- coding: utf-8 -*-
# 星算语 fromzero · 统一 CI 入口
# 一条命令验证整条引擎链全绿:
#   [1] P3 多源头前端汇合 (七前端 → 同字节码)
#   [2] P2/P3 训练VM (字节码驱动训练 + 五项golden + 确定性并行)
import os, subprocess, sys

BASE = r'E:\StarSnow_Home\languages\fromzero'
PY = sys.executable

STAGES = [
    ('多源头前端汇合(七前端→同字节码)', ' _multi_source_test.py'),
    ('训练VM golden 自动验收',            ' _vm_golden_test.py'),
]

print('=' * 60)
print('  星算语 fromzero · 统一 CI')
print('=' * 60)
all_ok = True
for title, script in STAGES:
    print(f'\n▶ {title}')
    r = subprocess.run([PY, os.path.join(BASE, script.strip())],
                       capture_output=True, text=True, encoding='utf-8')
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr.strip():
        print('  [stderr] ' + r.stderr.strip()[:500])
    ok = (r.returncode == 0)
    all_ok = all_ok and ok
    print(f'  → 阶段{"通过" if ok else "失败"} (exit={r.returncode})')

print('\n' + '=' * 60)
print(f'  CI 总结果: {"全部通过 ✓ 引擎链焊死" if all_ok else "存在失败 ✗ 需修复"}')
print('=' * 60)
sys.exit(0 if all_ok else 1)
