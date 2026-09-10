# -*- coding: utf-8 -*-
# 星算语 fromzero · stdlib 字节码执行器 (Python 参考实现)
# 读 v0.3 XL01 字节码 → 解释执行 stdlib 算子(18~40,255) → 输出可验证结果
# 解析模式沿用 c_core/xl_vm_train.c 的 read_xlbin: 头7B, 逐指令 4B(opcode+3B LE operand)
# v0.3 增量: 指令流遇 halt(255) 结束, 其后 const_len(u32 LE) + 常量段
import sys, os

# stdlib 算子 opcode(对齐 spec §4)
OP = {
    18: 'str_len', 19: 'str_concat', 20: 'str_find', 21: 'str_sub', 22: 'str_split',
    23: 'arr_new', 24: 'arr_idx', 25: 'arr_push', 26: 'arr_len', 27: 'arr_iter_mark',
    28: 'io_print_str', 29: 'io_print_reg', 30: 'io_read_file', 31: 'io_write_file',
    32: 'io_read_line', 33: 'mmap_read', 34: 'mmap_write', 35: 'bind_addr',
    36: 'jmp', 37: 'jmp_if_zero', 38: 'mark', 39: 'load_const', 40: 'io_print_rstr',
    255: 'halt',
}

def read_xlbin_v03(path):
    """返回 (header_bytes, insts, consts)。insts=[(op,operand)], consts=[bytes]。"""
    b = open(path, 'rb').read()
    if b[:4] != b'XL01':
        raise ValueError('魔数不是 XL01')
    ver = b[4:7]
    if ver != bytes([3, 0, 0]):
        raise ValueError(f'期望 v0.3=[3,0,0], 实际 {list(ver)}')
    off = 7
    insts = []
    while off + 4 <= len(b):
        op = b[off]
        operand = b[off+1] | (b[off+2] << 8) | (b[off+3] << 16)
        off += 4
        if op == 255:  # halt: 指令流结束
            break
        insts.append((op, operand))
    # halt 后: const_len(u32 LE) + const_block
    consts = []
    if off + 4 <= len(b):
        cl = b[off] | (b[off+1] << 8) | (b[off+2] << 16) | (b[off+3] << 24)
        off += 4
        cend = off + cl
        while off < cend:
            n = b[off] | (b[off+1] << 8) | (b[off+2] << 16) | (b[off+3] << 24)
            off += 4
            consts.append(b[off:off+n])
            off += n
    return ver, insts, consts

class VM:
    def __init__(self, consts):
        self.R = [0] * 16          # R0~R15, 值可为 int 或 bytes(句柄)
        self.consts = consts
        self.trace = []

    def run(self, insts):
        pc = 0
        while pc < len(insts):
            op, operand = insts[pc]
            self.trace.append(op)
            if op <= 17:
                # 既有训练算子: v0.3 执行器只 trace 透传, 不执行数值(分层不混)
                pc += 1
                continue
            name = OP.get(op, '?')
            if op == 18:    # str_len(const[k]) -> R0
                self.R[0] = len(self.consts[operand])
            elif op == 28:  # io_print_str(const[k])
                sys.stdout.write(self.consts[operand].decode('utf-8') + '\n')
            elif op == 29:  # io_print_reg(R0)
                sys.stdout.write(str(self.R[0]) + '\n')
            elif op == 30:  # io_read_file(const[k] 路径) -> R0
                p = self.consts[operand].decode('utf-8')
                self.R[0] = open(p, 'rb').read()
            elif op == 31:  # io_write_file(const[k] 路径), 内容=R0
                p = self.consts[operand].decode('utf-8')
                data = self.R[0]
                if isinstance(data, int):
                    data = str(data).encode('utf-8')
                open(p, 'wb').write(data)
            elif op == 39:  # load_const(const[k]) -> R0(句柄, 不拷贝)
                self.R[0] = self.consts[operand]
            elif op == 40:  # io_print_rstr(R0 内容句柄)
                data = self.R[0]
                if isinstance(data, int):
                    sys.stdout.write(str(data) + '\n')
                else:
                    sys.stdout.write(data.decode('utf-8') + '\n')
            elif op == 36:  # jmp
                pc = operand
                continue
            elif op == 37:  # jmp_if_zero
                if self.R[0] == 0:
                    pc = operand
                    continue
            elif op in (27, 38):  # mark / iter_mark: 仅 trace
                pass
            else:
                # 本最小执行器未实现的 stdlib 算子: 明确报错而非静默
                raise NotImplementedError(f'opcode {op} ({name}) 未在最小执行器实现')
            pc += 1

def main():
    xl = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'stdlib_demo.xlbin')
    ver, insts, consts = read_xlbin_v03(xl)
    print(f'== 星算语 stdlib VM == 版本={list(ver)} 指令={len(insts)} 常量段={len(consts)}条')
    vm = VM(consts)
    vm.run(insts)
    print(f'== 执行 trace 算子序列: {vm.trace} ==')

if __name__ == '__main__':
    main()
