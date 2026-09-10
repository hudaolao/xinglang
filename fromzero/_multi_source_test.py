# -*- coding: utf-8 -*-
# 星算语 fromzero · P3 多源头前端汇合 Golden 测试
# 铁则: 同一逻辑程序 → 各前端字节码哈希一致(汇合判定)
import hashlib, os, subprocess, sys

BASE = r'E:\StarSnow_Home\languages\fromzero'
results = []

def check(name, cond, detail=''):
    results.append((name, cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  <- {detail}' if detail else ''))

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]

print('== 星算语 fromzero · P3 多源头前端汇合 ==')

# 前端清单: 参考前端 + 六个语言源头
FRONTS = {
    'cs_front':   ('cs_front',   'out/sample.xlbin', False),  # C# 参考前端(需 dotnet, 用已有产物)
    'lisp':       ('lisp_front', 'out/lisp.xlbin',   True),
    'apl':        ('apl_front',  'out/apl.xlbin',    True),
    'haskell':    ('haskell_front', 'out/haskell.xlbin', True),
    'erlang':     ('erlang_front',  'out/erlang.xlbin',  True),
    'qsharp':     ('qsharp_front',  'out/qsharp.xlbin',  True),
    'jlang':      ('jlang_front',   'out/jlang.xlbin',   True),
}

# 1. 重新生成可运行前端产物
for name, (front, out, runnable) in FRONTS.items():
    if not runnable:
        continue
    r = subprocess.run([sys.executable, os.path.join(BASE, front, front + '.py')],
                       capture_output=True, text=True, encoding='utf-8')
    check(f'{name} 生成字节码', r.returncode == 0 and os.path.exists(os.path.join(BASE, front, out)))

# 2. 七前端字节码哈希一致(汇合铁则)
hs = {}
for name, (front, out, _) in FRONTS.items():
    p = os.path.join(BASE, front, out)
    if os.path.exists(p):
        hs[name] = sha(p)
if len(hs) == len(FRONTS):
    check(f'{len(FRONTS)}前端字节码哈希一致', len(set(hs.values())) == 1, str(hs))
    check('汇合哈希=6e7600a5f7a6dae6', hs['cs_front'] == '6e7600a5f7a6dae6', hs['cs_front'])
else:
    check(f'{len(FRONTS)}前端字节码文件齐全', len(hs) == len(FRONTS), f'只有 {list(hs.keys())}')
xl = os.path.join(BASE, 'cs_front', 'out', 'sample.xlbin')
if os.path.exists(xl):
    raw = open(xl, 'rb').read()
    check('XL01 魔数', raw[:4] == b'XL01')
    check('三符版本头', raw[4:7] == bytes([1, 1, 1]))
    check('指令数为7(7头字节+7条×4字节)', len(raw) == 7 + 7 * 4)  # 4魔数+3版本+7*(1opcode+3operand)

total = len(results); passed = sum(1 for _, c in results if c)
print(f'\n== 结果: {passed}/{total} 通过 ==')
sys.exit(0 if passed == total else 1)
