# -*- coding: utf-8 -*-
# 修正 p2_mem_depth.c / p2_mem_depth_pos.c 的采样逻辑
import io

FILES = [
    r'D:\StarSnow_Home\languages\fromzero\c_core\p2_mem_depth.c',
    r'D:\StarSnow_Home\languages\fromzero\c_core\p2_mem_depth_pos.c',
]

for f in FILES:
    s = io.open(f, encoding='utf-8').read()
    s = s.replace('static int S[L];', 'static int S[L+1];')
    s = s.replace('for(int t=K;t<L;t++) S[t]=S[t-K];', 'for(int t=K;t<L+1;t++) S[t]=S[t-K];')
    s = s.replace(
        'for(int t=0;t<L-1;t++){ float p[V]; softmax_row(&LG[t*V],p,V);',
        'for(int t=0;t<L;t++){ float p[V]; softmax_row(&LG[t*V],p,V);')
    s = s.replace('return loss/(L-1);', 'return loss/L;')

    # hit_train 覆盖全部 L 个位置
    old_ht = '''    for(int t=0;t<L-1;t++){ int pred=0; float mx=LG[t*V];
        for(int j=1;j<V;j++) if(LG[t*V+j]>mx){mx=LG[t*V+j];pred=j;}
        if(pred==S[t+1]) c++; }
    return c;'''
    new_ht = '''    for(int t=0;t<L;t++){ int pred=0; float mx=LG[t*V];
        for(int j=1;j<V;j++) if(LG[t*V+j]>mx){mx=LG[t*V+j];pred=j;}
        if(pred==S[t+1]) c++; }
    return c;'''
    assert old_ht in s, 'hit_train block not found in ' + f
    s = s.replace(old_ht, new_ht)

    # auto_reg 整体替换: t=L-1 预测输入外的 S[L]
    old_ar = '''static int auto_reg(void){
    /* 用前 K 个真实 seed 填满 L,滚动预测,期望=周期延续 */
    int buf[L]; for(int i=0;i<L;i++) buf[i]=S[i%K];
    int expect[L]; for(int i=0;i<L;i++) expect[i]=S[(i+1)%K];
    int ok=0, steps=10;
    for(int step=0;step<steps;step++){
        for(int i=0;i<L;i++) S[i]=buf[i];
        forward();
        int pred=0; float mx=LG[(L-2)*V]; for(int j=1;j<V;j++) if(LG[(L-2)*V+j]>mx){mx=LG[(L-2)*V+j];pred=j;}
        if(pred==expect[(L-1+step)%K]) ok++;
        for(int i=0;i<L-1;i++) buf[i]=buf[i+1];
        buf[L-1]=pred;
    }
    gen_seq();
    return ok;
}'''
    new_ar = '''static int auto_reg(void){
    /* 自回归生成: 输入前 L 个真实值, t=L-1 预测输入外的 S[L](真正未知) */
    int buf[L]; for(int i=0;i<L;i++) buf[i]=S[i];
    int ok=0, steps=10;
    for(int step=0;step<steps;step++){
        for(int i=0;i<L;i++) S[i]=buf[i];
        forward();
        int pred=0; float mx=LG[(L-1)*V]; for(int j=1;j<V;j++) if(LG[(L-1)*V+j]>mx){mx=LG[(L-1)*V+j];pred=j;}
        if(pred==S[(L+step)%K]) ok++;
        for(int i=0;i<L-1;i++) buf[i]=buf[i+1];
        buf[L-1]=pred;
    }
    gen_seq();
    return ok;
}'''
    assert old_ar in s, 'auto_reg block not found in ' + f
    s = s.replace(old_ar, new_ar)

    # main 打印命中分母 L-1 -> L
    old_pr = 'k, best, best_ep, h, L-1, ar,'
    new_pr = 'k, best, best_ep, h, L, ar,'
    assert old_pr in s, 'print block not found in ' + f
    s = s.replace(old_pr, new_pr)

    io.open(f, 'w', encoding='utf-8', newline='\n').write(s)
    print('patched', f)
