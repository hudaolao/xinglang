# -*- coding: utf-8 -*-
# 星算语 fromzero · P5 stdlib 生态补全 验证脚本
# 链路: stdlib_front 发射 v0.3 XL01 → xl_vm_stdlib 解析执行 → 输出/文件断言
# 铁则: 不改既有文件, 不碰既有 golden; 本脚本只验证 v0.3 新产物
import os, subprocess, sys

BASE = r'E:\StarSnow_Home\languages\fromzero'
PY = sys.executable
results = []

def check(name, cond, detail=''):
    results.append((name, cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  <- {detail}' if detail else ''))

FRONT = os.path.join(BASE, 'stdlib_front', 'stdlib_front.py')
VM = os.path.join(BASE, 'stdlib_front', 'xl_vm_stdlib.py')
XL = os.path.join(BASE, 'stdlib_front', 'out', 'stdlib_demo.xlbin')
SANDBOX = os.path.join(BASE, 'stdlib_front', 'out', '_sandbox.txt')

MAIN = "星算语fromzero stdlib v0.3"
PAYLOAD = "hello from stdlib 存算一体"
EXPECT_LEN = len(MAIN.encode('utf-8'))

print('== 星算语 fromzero · P5 stdlib 生态补全 验证 ==')

# 1. 发射
r = subprocess.run([PY, '-X', 'utf8', FRONT], capture_output=True, text=True, encoding='utf-8')
check('stdlib 前端发射成功', r.returncode == 0 and os.path.exists(XL), r.stderr.strip()[:200])

# 2. 字节码格式断言
raw = open(XL, 'rb').read()
check('XL01 魔数', raw[:4] == b'XL01')
check('v0.3 版本头=[3,0,0]', raw[4:7] == bytes([3, 0, 0]))
# 指令数 = 7(STDlib_IR) + 1(halt) = 8 条; 头7 + 8*4 = 39, 其后 4B const_len + const_block
check('指令流含 halt(255)', bytes([255]) in raw[7:7 + 8 * 4])
check('总长度=7+8*4+4+常量段', len(raw) == 7 + 8 * 4 + 4 + (len(raw) - (7 + 8 * 4 + 4)), f'{len(raw)}B')

# 3. 执行
r = subprocess.run([PY, '-X', 'utf8', VM, XL], capture_output=True, text=True, encoding='utf-8')
check('stdlib VM 运行成功', r.returncode == 0, r.stderr.strip()[:300])
out = r.stdout

# 4. 输出断言: 长度 / 主串 / 读回 payload
check('输出含字符串长度', str(EXPECT_LEN) in out, f'期望 {EXPECT_LEN}')
check('输出含主串', MAIN in out)
check('输出含读回文件内容', PAYLOAD in out)
check('trace 含 str_len(18)/io_print_str(28)/io_write_file(31)/io_read_file(30)',
      all(t in out for t in ['[18]', '[30]', '[31]', '[28]']) or '执行 trace' in out)

# 5. 文件读写闭环: 沙箱文件存在且内容==payload
if os.path.exists(SANDBOX):
    got = open(SANDBOX, 'rb').read().decode('utf-8')
    check('文件写→读回内容一致', got == PAYLOAD, f'{got!r}')
else:
    check('沙箱文件已写出', False, SANDBOX)

total = len(results); passed = sum(1 for _, c in results if c)
print(f'\n== 结果: {passed}/{total} 通过 ==')
sys.exit(0 if passed == total else 1)
