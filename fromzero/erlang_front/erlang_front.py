# -*- coding: utf-8 -*-
# 星算语 · Erlang 前端(消息即计算源头) → XL01 字节码
# P3 多源头汇合: 同一逻辑程序, Erlang 风格 → 与其它前端字节码哈希一致
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _kit import run

NAME = 'erlang'
SRC = r"""
%% 星算语 Erlang 前端 · 消息即计算源头 · 同一逻辑程序
loc(x).
transform().
matmul(x,64).
layernorm(y).
qreg(42).
flow(z,32).
adamw(1).
"""

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', NAME + '.xlbin')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run(NAME, SRC, out)

main()
