# -*- coding: utf-8 -*-
# 星算语 · stdlib 前端 → v0.3 XL01 字节码
# P5 = 把星算语从「能训练」推进到「能干活」
# 对齐: spec/xinglang_v03_stdlib.md (opcode 18~40 + halt255, 常量段格式)
# 发射模式沿用 train_front/train_front.py 的 emit_xl01: 头7B + 4B定长指令, 不发明新管线。
import os, hashlib

# stdlib 算子 opcode 速查(对齐 v0.3 spec §4)
# 18=str_len 19=str_concat 20=str_find 21=str_sub 22=str_split
# 23=arr_new 24=arr_idx 25=arr_push 26=arr_len 27=arr_iter_mark
# 28=io_print_str 29=io_print_reg 30=io_read_file 31=io_write_file 32=io_read_line
# 33=mmap_read 34=mmap_write 35=bind_addr 36=jmp 37=jmp_if_zero 38=mark
# 39=load_const 40=io_print_rstr 255=halt

def pack12(a, b):
    """双操作数 operand: (A:12b << 12) | (B:12b), 24bit。"""
    return ((a & 0xFFF) << 12) | (b & 0xFFF)

def emit_xl01_v03(insts, consts, out_path):
    """发射 v0.3 XL01 字节码。
    insts : [(opcode, operand, 注释), ...]  不含 halt(自动追加)
    consts: [bytes, ...] 常量段条目, 编号从 0 起
    布局: "XL01" + [3,0,0] + 指令流 + halt(255,0) + const_len(u32 LE) + const_block
    """
    with open(out_path, 'wb') as w:
        w.write(b'XL01')
        w.write(bytes([3, 0, 0]))          # v0.3 stdlib 三符版本头
        for op, operand, _ in insts:
            w.write(bytes([op]))
            w.write(bytes([operand & 0xFF, (operand >> 8) & 0xFF, (operand >> 16) & 0xFF]))
        # halt: 标记指令流结束
        w.write(bytes([255, 0, 0, 0]))
        # 常量段: (u32 LE len)(bytes) 逐条
        cb = bytearray()
        for c in consts:
            n = len(c)
            cb += bytes([n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF])
            cb += c
        cl = len(cb)
        w.write(bytes([cl & 0xFF, (cl >> 8) & 0xFF, (cl >> 16) & 0xFF, (cl >> 24) & 0xFF]))
        w.write(bytes(cb))

# 最小可运行程序: 字符串长度 + 打印 + 文件写 + 文件读 + 打印读回
# 常量段
C_MAIN   = "星算语fromzero stdlib v0.3".encode('utf-8')   # 条目0: 主串
C_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', '_sandbox.txt').encode('utf-8')  # 条目1: 沙箱文件路径
C_PAYLOAD = "hello from stdlib 存算一体".encode('utf-8')   # 条目2: 写文件内容

STDlib_IR = [
    (18, 0,       'str_len(const[0])  -> R0 = UTF-8 字节长度'),
    (29, 0,       'io_print_reg      -> 打印 R0(长度)'),
    (28, 0,       'io_print_str(0)   -> 打印主串'),
    (39, 2,       'load_const(2)     -> R0 = 写文件内容句柄'),
    (31, 1,       'io_write_file(1)  -> R0 内容写入沙箱文件'),
    (30, 1,       'io_read_file(1)   -> R0 = 读回内容句柄'),
    (40, 0,       'io_print_rstr     -> 打印读回内容'),
]

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'stdlib_demo.xlbin')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    emit_xl01_v03(STDlib_IR, [C_MAIN, C_PATH, C_PAYLOAD], out)
    raw = open(out, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()[:16]
    print(f'stdlib 前端: {len(STDlib_IR)} 条指令 + halt → {out}')
    for op, operand, note in STDlib_IR:
        print(f'   op={op:>3} operand={operand:<5} {note}')
    print(f'   常量段: 3 条 (主串{len(C_MAIN)}B / 路径 / 负载{len(C_PAYLOAD)}B)')
    print(f'   字节码总长: {len(raw)} 字节')
    print(f'   v0.3 字节码哈希: {h}')

if __name__ == '__main__':
    main()
