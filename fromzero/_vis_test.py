# -*- coding: utf-8 -*-
# 星算语 fromzero · P5 可视化/IO 最小可运行验证
# 验证链: vis_front.py 发射 XL01 → vis_render.py 解析+渲染 → 断言 PPM 尺寸/像素
# 铁则: 纯标准库, 无外部依赖; 既有 golden 哈希不动
import os, subprocess, sys

BASE = r'E:\StarSnow_Home\languages\fromzero'
VIS = os.path.join(BASE, 'vis_front')
PY = sys.executable

results = []
def check(name, cond, detail=''):
    results.append((name, cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  <- {detail}' if detail else ''))

def parse_ppm_p6(path):
    """解析 P6 PPM: 返回 (w, h, bytes_pixels)。像素行优先 RGB。"""
    raw = open(path, 'rb').read()
    # 解析三个 ASCII 头 token: 魔数 P6, w, h, maxval
    # 简单状态机: 跳过注释(#...)和空白
    toks = []
    i = 0
    while len(toks) < 4:
        # 跳过空白
        while i < len(raw) and raw[i:i+1].isspace():
            i += 1
        # 跳过注释
        if i < len(raw) and raw[i:i+1] == b'#':
            while i < len(raw) and raw[i:i+1] != b'\n':
                i += 1
            continue
        # 读 token
        j = i
        while j < len(raw) and not raw[j:j+1].isspace():
            j += 1
        toks.append(raw[i:j].decode('ascii'))
        i = j
    magic, w, h, maxval = toks[0], int(toks[1]), int(toks[2]), int(toks[3])
    # i 指向最后一个头 token 后; 跳过它后面的空白(通常一个换行)
    while i < len(raw) and raw[i:i+1].isspace():
        i += 1
    pixels = raw[i:]
    return magic, w, h, maxval, pixels

def pixel_at(pixels, w, h, x, y):
    idx = (y * w + x) * 3
    return pixels[idx], pixels[idx+1], pixels[idx+2]

print('== 星算语 fromzero · P5 可视化/IO 最小验证 ==')

# 1. 跑前端发射字节码
r = subprocess.run([PY, os.path.join(VIS, 'vis_front.py')],
                   capture_output=True, text=True, encoding='utf-8')
check('vis_front.py 发射成功', r.returncode == 0, r.stderr[:200])
if r.stdout:
    print('  ' + r.stdout.rstrip().replace('\n', '\n  '))

xlbin = os.path.join(VIS, 'out', 'vis_demo.xlbin')
check('vis_demo.xlbin 存在', os.path.exists(xlbin), xlbin)

# 2. 字节码头检查
raw = open(xlbin, 'rb').read()
check('XL01 魔数', raw[:4] == b'XL01', raw[:4])
check('v0.3 版本头 [3,0,0]', list(raw[4:7]) == [3, 0, 0], list(raw[4:7]))
check('指令数合理(头7 + N*4)', (len(raw) - 7) % 4 == 0, f'总长 {len(raw)}')

# 3. 跑渲染器
r = subprocess.run([PY, os.path.join(VIS, 'vis_render.py')],
                   capture_output=True, text=True, encoding='utf-8')
check('vis_render.py 渲染成功', r.returncode == 0, r.stderr[:300])
if r.stdout:
    print('  ' + r.stdout.rstrip().replace('\n', '\n  '))

ppm = os.path.join(VIS, 'out', 'vis_demo.ppm')
check('vis_demo.ppm 存在', os.path.exists(ppm), ppm)

# 4. PPM 格式与像素断言
magic, w, h, maxval, pixels = parse_ppm_p6(ppm)
check('PPM 魔数 P6', magic == 'P6', magic)
check('画布宽=128', w == 128, w)
check('画布高=64', h == 64, h)
check('最大亮度=255', maxval == 255, maxval)
check('像素数据长度 = 128*64*3', len(pixels) == 128*64*3, len(pixels))

# 关键像素: 渐变公式 (255-4y, 128, 4y)
check('(0,0) 顶点红 (255,128,0)',
      pixel_at(pixels, w, h, 0, 0) == (255, 128, 0),
      pixel_at(pixels, w, h, 0, 0))
check('(0,63) 底点蓝渐变 (3,128,252)',
      pixel_at(pixels, w, h, 0, 63) == (3, 128, 252),
      pixel_at(pixels, w, h, 0, 63))
# 白色实心矩形 (40..88, 20..44), 中心 (64,32)
check('矩形中心 (64,32) = 白 (255,255,255)',
      pixel_at(pixels, w, h, 64, 32) == (255, 255, 255),
      pixel_at(pixels, w, h, 64, 32))
# 绿色标记点 (100,10)
check('绿点 (100,10) = (0,255,0)',
      pixel_at(pixels, w, h, 100, 10) == (0, 255, 0),
      pixel_at(pixels, w, h, 100, 10))
# 矩形外 (120,60) 仍在渐变上: y=60 → (255-240=15, 128, 240)
check('矩形外 (120,60) = 渐变 (15,128,240)',
      pixel_at(pixels, w, h, 120, 60) == (15, 128, 240),
      pixel_at(pixels, w, h, 120, 60))

total = len(results); passed = sum(1 for _, c in results if c)
print(f'\n== 结果: {passed}/{total} 通过 ==')
sys.exit(0 if passed == total else 1)
