# -*- coding: utf-8 -*-
# 星算语 fromzero · v0.3 量纲类型系统 最小可运行验证
# 用法: python -X utf8 test_dim.py
# 断言:
#   1) 正确量纲用例 test_dim_ok.xxl 编译通过 (退出码 0)
#   2) 量纲不匹配用例 test_dim_bad.xxl 编译失败 (退出码 !=0) 且报错带行号
#   3) 既有 SAMPLE 编译产物 out.xlbin 字节级不变 (MD5 = 596cf4355075cafdd0cc03c703ac0cda)
import os, subprocess, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "target", "debug", "xl_compiler.exe")
OUT = os.path.join(HERE, "out.xlbin")
BASELINE_MD5 = "596cf4355075cafdd0cc03c703ac0cda"

def run(src):
    return subprocess.run([EXE, src], capture_output=True, text=True, encoding="utf-8")

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

fails = []
def check(name, cond):
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    if not cond:
        fails.append(name)

print("== v0.3 量纲类型系统 验证 ==")

# 1) 正确用例
r = run(os.path.join(HERE, "test_dim_ok.xxl"))
print("[ok 用例] exit=%s" % r.returncode)
check("正确量纲用例退出码 0", r.returncode == 0)
check("正确量纲用例输出含'量纲门禁通过'", "量纲门禁通过" in r.stdout)

# 2) 错误用例
r = run(os.path.join(HERE, "test_dim_bad.xxl"))
print("[bad 用例] exit=%s stderr=%s" % (r.returncode, r.stderr.strip()))
check("量纲不匹配用例退出码非 0", r.returncode != 0)
check("报错带行号 [第 4 行]", "[第 4 行]" in r.stderr)
check("报错点明 GB + MB 不匹配", "GB + MB" in r.stderr and "量纲不匹配" in r.stderr)

# 3) 既有 SAMPLE 产物不变 (回归保护)
subprocess.run([EXE], capture_output=True)  # 无参 → SAMPLE
h = md5(OUT)
print("[回归] SAMPLE out.xlbin md5=%s" % h)
check("既有 SAMPLE 字节码未变", h == BASELINE_MD5)

print("-" * 48)
if fails:
    print("  结果: 有失败 ✗ : %s" % fails)
    sys.exit(1)
print("  结果: 全部通过 ✓ 量纲类型系统焊死")
