# -*- coding: utf-8 -*-
# 星算语 · 可视化 XL01 渲染器 / 验证器
# 解析 v0.3 可视化字节码, 执行最小渲染管线:
#   vcanvas_new → vpen_color/vpen_move/vfill_rect → vpix_at → vrender_ppm
# 产出 P6 二进制 PPM, 纯标准库, 无外部依赖。
# 对齐: spec/xinglang_v03_visual_io.md
import os, sys, struct

# ---- v0.3 可视化算子 opcode (与 spec 对齐) ----
OP = {
    48: 'vcanvas_new', 49: 'vpen_move', 50: 'vpen_color',
    51: 'vpix_dot', 52: 'vpix_at', 53: 'vpix_read',
    54: 'vfill_rect', 55: 'vfill_all', 56: 'vload_image',
    57: 'vscale', 58: 'vrotate', 59: 'vtext_draw',
    60: 'vrender_ppm', 61: 'vrender_png', 62: 'vflush',
}

def unpack_xy(operand):
    y = operand & 0xFFF
    x = (operand >> 12) & 0xFFF
    return x, y

def unpack_rgb(operand):
    b = operand & 0xFF
    g = (operand >> 8) & 0xFF
    r = (operand >> 16) & 0xFF
    return r, g, b

class Canvas:
    """RGB 帧缓冲: W x H, 行优先, (0,0) 左上。"""
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.buf = bytearray(w * h * 3)  # 全 0 = 黑

    def _idx(self, x, y):
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise ValueError(f'像素越界: ({x},{y}) 不在 {self.w}x{self.h} 内')
        return (y * self.w + x) * 3

    def put(self, x, y, rgb):
        i = self._idx(x, y)
        self.buf[i] = rgb[0]
        self.buf[i+1] = rgb[1]
        self.buf[i+2] = rgb[2]

    def get(self, x, y):
        i = self._idx(x, y)
        return self.buf[i], self.buf[i+1], self.buf[i+2]

    def fill_rect(self, x0, y0, x1, y1, rgb):
        # 左上角必须 <= 右下角(确定性硬规则, 不自动交换)
        if x1 < x0 or y1 < y0:
            raise ValueError(f'vfill_rect 反转: ({x0},{y0})→({x1},{y1})')
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.put(x, y, rgb)

    def fill_all(self, rgb):
        for y in range(self.h):
            for x in range(self.w):
                self.put(x, y, rgb)

    def to_ppm_p6(self):
        header = f'P6\n{self.w} {self.h}\n255\n'.encode('ascii')
        return header + bytes(self.buf)

def parse_xl01(path):
    raw = open(path, 'rb').read()
    if raw[:4] != b'XL01':
        raise ValueError(f'魔数不对: {raw[:4]!r}')
    ver = raw[4:7]
    body = raw[7:]
    if len(body) % 4 != 0:
        raise ValueError(f'指令体长度 {len(body)} 不是 4 的倍数')
    insts = []
    for off in range(0, len(body), 4):
        op = body[off]
        operand = body[off+1] | (body[off+2] << 8) | (body[off+3] << 16)
        insts.append((op, operand))
    return ver, insts

def render(xlbin_path, out_dir):
    ver, insts = parse_xl01(xlbin_path)
    print(f'[render] 版本头: {list(ver)}, 指令数: {len(insts)}')

    canvas = None
    pen_x, pen_y = 0, 0
    pen_rgb = (0, 0, 0)
    trace = []

    for pc, (op, operand) in enumerate(insts):
        name = OP.get(op, f'unknown({op})')
        if op == 48:  # vcanvas_new
            w, h = unpack_xy(operand)
            canvas = Canvas(w, h)
            pen_x, pen_y = 0, 0
            pen_rgb = (0, 0, 0)
            trace.append(f'pc={pc} vcanvas_new {w}x{h}')
        elif op == 49:  # vpen_move
            pen_x, pen_y = unpack_xy(operand)
        elif op == 50:  # vpen_color
            pen_rgb = unpack_rgb(operand)
        elif op == 51:  # vpix_dot
            canvas.put(pen_x, pen_y, pen_rgb)
        elif op == 52:  # vpix_at
            x, y = unpack_xy(operand)
            canvas.put(x, y, pen_rgb)
        elif op == 53:  # vpix_read
            x, y = unpack_xy(operand)
            rgb = canvas.get(x, y)
            trace.append(f'pc={pc} vpix_read({x},{y}) = {rgb}')
        elif op == 54:  # vfill_rect
            x0, y0 = unpack_xy(operand)
            canvas.fill_rect(x0, y0, pen_x, pen_y, pen_rgb)
        elif op == 55:  # vfill_all
            canvas.fill_all(pen_rgb)
        elif op == 56:  # vload_image (接口标记)
            trace.append(f'pc={pc} vload_image path_id={(operand>>8)&0xFFFF} [接口冻结]')
        elif op in (57, 58, 59):  # vscale/vrotate/vtext_draw (接口标记)
            trace.append(f'pc={pc} {name} operand={operand} [接口冻结]')
        elif op == 60:  # vrender_ppm
            out_id = (operand >> 8) & 0xFFFF
            fmt = operand & 0xFF
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'vis_demo.ppm')
            if fmt == 0:
                data = canvas.to_ppm_p6()
            else:
                # P3 文本镜像
                lines = [f'P3', f'{canvas.w} {canvas.h}', '255']
                for i in range(canvas.w * canvas.h):
                    r = canvas.buf[i*3]; g = canvas.buf[i*3+1]; b = canvas.buf[i*3+2]
                    lines.append(f'{r} {g} {b}')
                data = '\n'.join(lines).encode('ascii')
            with open(out_path, 'wb') as f:
                f.write(data)
            trace.append(f'pc={pc} vrender_ppm out_id={out_id} fmt={fmt} -> {out_path} ({len(data)}B)')
        elif op == 61:  # vrender_png (接口标记)
            trace.append(f'pc={pc} vrender_png [接口冻结, 不实现]')
        elif op == 62:  # vflush
            trace.append(f'pc={pc} vflush')
        else:
            trace.append(f'pc={pc} 未知算子 op={op} operand={operand} (跳过)')

    for t in trace:
        print('   ' + t)
    return canvas, trace

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    xlbin = os.path.join(here, 'out', 'vis_demo.xlbin')
    out_dir = os.path.join(here, 'out')
    if not os.path.exists(xlbin):
        print(f'[render] 找不到 {xlbin}, 请先跑 vis_front.py')
        sys.exit(1)
    canvas, trace = render(xlbin, out_dir)
    print(f'[render] 完成: {canvas.w}x{canvas.h} 画布, PPM 已写出')

if __name__ == '__main__':
    main()
