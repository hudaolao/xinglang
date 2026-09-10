# -*- coding: utf-8 -*-
# 星算语 fromzero · 训练VM 五项哈希 golden 自动验收
# 跑字节码执行器(1线程), 对比 golden\train_vm_golden.json 五项哈希
import json, os, subprocess, sys

BASE = r'E:\StarSnow_Home\languages\fromzero'
results = []

# OpenMP 运行时 DLL (libgomp-1.dll) 不在默认 PATH, 注入 mingw64\bin
os.environ['PATH'] = r'D:\StarSnow_Home\toolchain\msys64\mingw64\bin' + os.pathsep + os.environ.get('PATH', '')

def check(name, cond, detail=''):
    results.append((name, cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  <- {detail}' if detail else ''))

print('== 星算语 fromzero · 训练VM golden 验收 ==')

# 1. 跑执行器(单线程)
exe = os.path.join(BASE, 'c_core', 'xl_vm_train.exe')
ir = os.path.join(BASE, 'train_front', 'out', 'train_step.xlbin')
if not os.path.exists(exe):
    check('执行器存在', False, '需先 gcc 编译 xl_vm_train.c')
    print(f'\n== 结果: 0/{len(results)} =='); sys.exit(1)
out = subprocess.run([exe, ir, '1'], capture_output=True, text=True, encoding='utf-8').stdout
check('执行器运行成功', 'loss序列' in out)

# 2. 解析五项哈希
import re
got = {}
for name in ['字节码', '预算报告', '执行trace', '归约浮点', 'loss序列']:
    m = re.search(name + r':\s*([0-9a-f]{16})', out)
    got[name] = m.group(1) if m else None

# 3. 对比 golden
gp = os.path.join(BASE, 'golden', 'train_vm_golden.json')
if os.path.exists(gp):
    g = json.load(open(gp, encoding='utf-8'))['hashes']
    keymap = {'字节码': 'bytecode', '预算报告': 'budget', '执行trace': 'trace',
              '归约浮点': 'reduction', 'loss序列': 'loss_seq'}
    all_ok = True
    for cn, key in keymap.items():
        exp = g.get(key)
        act = got.get(cn)
        check(f'golden {cn}', act == exp, f'{act} vs {exp}')
        if act != exp:
            all_ok = False
else:
    check('golden文件', False, '缺失 train_vm_golden.json')
    all_ok = False

total = len(results); passed = sum(1 for _, c in results if c)
print(f'\n== 结果: {passed}/{total} 通过 ==')
sys.exit(0 if passed == total and all_ok else 1)
