# -*- coding: utf-8 -*-
# 星算语 · 可视化前端 → v0.3 可视化级 XL01 字节码
# P5 = 可视化/IO 表达补全: 把"画一张图"表达成统一 XL01 算子流
# 对齐: spec/xinglang_v03_visual_io.md (算子 48-62, 画布即位置, 读取即显示)
#
# 演示程序: 128x64 画布, 垂直渐变色带 + 白色实心矩形 + 绿色标记点, 渲染成 P6 PPM
import os, sys, hashlib

# ---- v0.3 可视化算子 opcode 表(与 spec 对齐, 48-62) ----
OP = {
    'vcanvas_new': 48, 'vpen_move': 49, 'vpen_color': 50,
    'vpix_dot': 51, 'vpix_at': 52, 'vpix_read': 53,
    'vfill_rect': 54, 'vfill_all': 55, 'vload_image': 56,
    'vscale': 57, 'vrotate': 58, 'vtext_draw': 59,
    'vrender_ppm': 60, 'vrender_png': 61, 'vflush': 62,
}

# ---- operand 打包工具(24-bit 小端) ----
def pack_xy(x, y):
    """X[11:0]<<12 | Y[11:0], 坐标各 12 bit (0..4095)"""
    assert 0 <= x < 4096 and 0 <= y < 4096
    return ((x & 0xFFF) << 12) | (y & 0xFFF)

def pack_rgb(r, g, b):
    """R[7:0]<<16 | G[7:0]<<8 | B[7:0]"""
    assert all(0 <= c < 256 for c in (r, g, b))
    return ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)

def pack_wl(w, h):
    """宽高: W[11:0]<<12 | H[11:0]"""
    return pack_xy(w, h)

# ---- 演示程序: 构造指令流 ----
W, H = 128, 64

def build_demo():
    insts = []  # (opcode, operand, 注释)
    # 1) 创建画布 128x64
    insts.append((OP['vcanvas_new'], pack_wl(W, H), f'vcanvas_new({W},{H})'))
    # 2) 背景清屏为黑(画布本就黑, 显式调用以覆盖 vfill_all 语义)
    insts.append((OP['vpen_color'], pack_rgb(0, 0, 0), 'pen_color(0,0,0)'))
    insts.append((OP['vfill_all'], 0, 'vfill_all 黑底'))
    # 3) 垂直渐变: 每行 y → (255-4y, 128, 4y)
    #    行首: 设色; vpen_move(W-1, y) 作右下角; vfill_rect(0, y) 作左上角
    for y in range(H):
        r = 255 - 4 * y
        g = 128
        b = 4 * y
        insts.append((OP['vpen_color'], pack_rgb(r, g, b), f'row y={y} color=({r},{g},{b})'))
        insts.append((OP['vpen_move'], pack_xy(W - 1, y), f'pen→({W-1},{y})'))
        insts.append((OP['vfill_rect'], pack_xy(0, y), f'fill rect (0,{y})→({W-1},{y})'))
    # 4) 白色实心矩形: x 40..88, y 20..44
    insts.append((OP['vpen_color'], pack_rgb(255, 255, 255), 'pen_color(255,255,255)'))
    insts.append((OP['vpen_move'], pack_xy(88, 44), 'pen→(88,44)'))
    insts.append((OP['vfill_rect'], pack_xy(40, 20), 'fill rect (40,20)→(88,44)'))
    # 5) 绿色标记点 (100,10) — 验证用
    insts.append((OP['vpen_color'], pack_rgb(0, 255, 0), 'pen_color(0,255,0)'))
    insts.append((OP['vpix_at'], pack_xy(100, 10), 'pix_at(100,10) 绿点'))
    # 6) 读像素 trace(不改画布) — 演示 vpix_read
    insts.append((OP['vpix_read'], pack_xy(64, 32), 'pix_read(64,32) trace'))
    # 7) 渲染为 P6 PPM (out_id=0, fmt=0)
    insts.append((OP['vrender_ppm'], (0 << 8) | 0, 'vrender_ppm(out_id=0, fmt=P6)'))
    insts.append((OP['vflush'], 0, 'vflush'))
    return insts

# ---- 标准 XL01 发射管线(对齐 _kit.py / train_front.py) ----
def emit_xl03(insts, out_path):
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([3, 0, 0]))   # v0.3 三符版本(可视化/IO)
        for op, operand, _ in insts:
            w.write(bytes([op & 0xFF]))
            w.write(bytes([operand & 0xFF,
                           (operand >> 8) & 0xFF,
                           (operand >> 16) & 0xFF]))

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'out')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'vis_demo.xlbin')

    insts = build_demo()
    emit_xl03(insts, out)
    raw = open(out, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()[:16]

    print(f'可视化前端: {len(insts)} 条指令 → {out}')
    print(f'   画布: {W}x{H}  (RGB)')
    # 只打印首尾几条, 渐变行太长不全打
    preview = insts[:4] + [('...', None, f'... {H} 行渐变 ...')] + insts[-5:]
    for item in preview:
        if item[0] == '...':
            print(f'   {item[2]}')
        else:
            op, operand, note = item
            print(f'   op={op:>2} operand={operand:<8} {note}')
    print(f'   字节码总长: {len(raw)} 字节  (头7 + {len(insts)}*4)')
    print(f'   可视化IR哈希: {h}')

if __name__ == '__main__':
    main()
