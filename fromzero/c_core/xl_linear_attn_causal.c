// 星算语 fromzero · P2 因果单头线性注意力语言模型
// 因果(state 增量,只看过去): state_t=state_{t-1}+k_t⊗v_t; z_t=z_{t-1}+k_t
//   y_t=(state_t^T q_t)/(z_t·q_t+eps)  —— 内存 O(d²),无限上下文
// 反向:按时间 t=L-1..0 回推,维护 state 梯度 G 与 z 梯度 Gz
// 确定性:合同PCG32 + 固定归约序 + 解析反向 + 梯度裁剪 + adamw
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 32
#define D 32
#define L 16
#define EPOCHS 200

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static int S[L];
static void gen_seq(void){ int motif[4]={3,7,1,9}; for(int t=0;t<L;t++) S[t]=motif[t%4]; }

static float E[V*D], Wq[D*D], Wk[D*D], Wv[D*D], Wo[V*D];
static float mE[V*D],mQ[D*D],mK[D*D],mV[D*D],mO[V*D];
static float vE[V*D],vQ[D*D],vK[D*D],vV[D*D],vO[V*D];

static float X[L*D], Q[L*D], K[L*D], VV[L*D], Y[L*D], LG[L*V];
static float STATE[L*D*D], Z[L*D], NUM[L*D], DEN[L];
static float RQ[L], RK[L];   /* RMS 归一化因子 */
static float SDEN[L];        /* den 内部量 z·q' 的符号 */

static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
static float elu(float x){ return x>0 ? x : expm1f(x); }
static float elu_p(float x){ return x>0 ? 1.0f : expf(x); }

static void forward(void){
    float eps=1e-3f;   /* den 兜底,防 z·q→0 放大梯度 */
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) X[t*D+d]=E[S[t]*D+d];
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        double sq=0,sk=0,sv=0;
        for(int e=0;e<D;e++){ sq+=(double)Wq[d*D+e]*X[t*D+e]; sk+=(double)Wk[d*D+e]*X[t*D+e]; sv+=(double)Wv[d*D+e]*X[t*D+e]; }
        Q[t*D+d]=(float)sq; K[t*D+d]=(float)sk; VV[t*D+d]=(float)sv;
    }
    /* RMS 归一化 q,k: q'=q/rms(q),k'=k/rms(k), ||q'||=||k'||=sqrt(D), den 稳定 */
    for(int t=0;t<L;t++){
        double q2=0,k2=0;
        for(int d=0;d<D;d++){ q2+=(double)Q[t*D+d]*Q[t*D+d]; k2+=(double)K[t*D+d]*K[t*D+d]; }
        double rq=sqrt(q2/D)+1e-6, rk=sqrt(k2/D)+1e-6;
        RQ[t]=(float)rq; RK[t]=(float)rk;
        for(int d=0;d<D;d++){ Q[t*D+d]=(float)((double)Q[t*D+d]/rq); K[t*D+d]=(float)((double)K[t*D+d]/rk); }
    }
    float st[D*D]; for(int i=0;i<D*D;i++) st[i]=0;
    float z[D]; for(int i=0;i<D;i++) z[i]=0;
    for(int t=0;t<L;t++){
        /* state += elu核(k) ⊗ v: 用 k' = elu(k)+1 做键 */
        for(int i=0;i<D;i++){ float ki=elu(K[t*D+i])+1.0f;
            for(int j=0;j<D;j++) st[i*D+j]+=ki*VV[t*D+j];
            z[i]+=ki; }
        for(int i=0;i<D*D;i++) STATE[t*D*D+i]=st[i];
        for(int i=0;i<D;i++) Z[t*D+i]=z[i];
        for(int j=0;j<D;j++){ double s=0; for(int i=0;i<D;i++) s+=(double)st[i*D+j]*Q[t*D+i]; NUM[t*D+j]=(float)s; }
        /* den=|z·q'|+eps: 防 z·q' 为负导致 y 符号翻转(梯度错乱) */
        double s=0; for(int i=0;i<D;i++) s+=(double)z[i]*Q[t*D+i];
        double de=fabs(s)+eps;
        SDEN[t]=(float)(s>=0?1.0f:-1.0f);
        DEN[t]=(float)de;
        for(int j=0;j<D;j++) Y[t*D+j]=NUM[t*D+j]/DEN[t];
    }
    for(int t=0;t<L;t++) for(int j=0;j<V;j++){
        double s=0; for(int d=0;d<D;d++) s+=(double)Wo[j*D+d]*Y[t*D+d];
        LG[t*V+j]=(float)s;
    }
}

static double train_epoch(float lr, float b1, float b2, float eps_a, int ep){
    float eps=1e-6f;
    forward();
    static float dLG[L*V];
    double loss=0;
    for(int t=0;t<L-1;t++){ float p[V]; softmax_row(&LG[t*V],p,V);
        int target=S[t+1]; double q=(double)p[target]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++) dLG[t*V+j]=(j==target)?(p[j]-1.0f):p[j]; }

    static float dY[L*D], dWo[V*D], dWq[D*D], dWk[D*D], dWv[D*D], dE[V*D];
    static float dQ[L*D], dK[L*D], dV[L*D];
    /* 关键:static 数组跨调用保留值,必须每轮清零(否则梯度跨epoch累积,方向混合→波动) */
    for(int i=0;i<D*D;i++){ dWq[i]=0; dWk[i]=0; dWv[i]=0; }
    for(int i=0;i<V*D;i++) dE[i]=0;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){ double s=0;
        for(int j=0;j<V;j++) s+=(double)dLG[t*V+j]*Wo[j*D+d]; dY[t*D+d]=(float)s; }
    for(int j=0;j<V;j++) for(int d=0;d<D;d++){ double s=0;
        for(int t=0;t<L;t++) s+=(double)dLG[t*V+j]*Y[t*D+d]; dWo[j*D+d]=(float)s; }

    /* 因果线性注意力反向:时间回推,维护 G(state) 与 Gz(z) */
    static float G[D*D], Gz[D], dstate[D*D];
    for(int i=0;i<D*D;i++) G[i]=0;
    for(int i=0;i<D;i++) Gz[i]=0;
    for(int t=L-1;t>=0;t--){
        float dnum[D]; for(int j=0;j<D;j++) dnum[j]=dY[t*D+j]/DEN[t];
        double dden=0; for(int j=0;j<D;j++) dden+=(double)dY[t*D+j]*NUM[t*D+j]; dden=-dden/((double)DEN[t]*DEN[t]);
        dden*=(double)SDEN[t];   /* 对 s=z·q' 的梯度(经 |·| 链式) */
        /* dq_t: num 部分 + den 部分 */
        for(int i=0;i<D;i++){
            double dqi=(double)((float)(dden*Z[t*D+i]));
            for(int j=0;j<D;j++) dqi+=(double)dnum[j]*STATE[t*D*D+i*D+j];
            dQ[t*D+i]=(float)dqi;
        }
        /* Gz 加 den 贡献 */
        for(int i=0;i<D;i++) Gz[i]+=(float)(dden*Q[t*D+i]);
        /* dstate = G + dnum[j]*q_t[i] */
        for(int i=0;i<D;i++) for(int j=0;j<D;j++)
            dstate[i*D+j]=G[i*D+j]+dnum[j]*Q[t*D+i];
        /* dk_t = Gz[i] + Σ_j dstate[i][j]*v[j]; dv_t[j]=Σ_i dstate[i][j]*k'[i] */
        for(int i=0;i<D;i++){ double dk=Gz[i]; for(int j=0;j<D;j++) dk+=(double)dstate[i*D+j]*VV[t*D+j]; dK[t*D+i]=(float)dk; }
        for(int j=0;j<D;j++){ double dv=0; for(int i=0;i<D;i++) dv+=(double)dstate[i*D+j]*(elu(K[t*D+i])+1.0f); dV[t*D+j]=(float)dv; }
        /* G 传给 state_{t-1} */
        for(int i=0;i<D*D;i++) G[i]=dstate[i];
        /* Gz 保持,传给 z_{t-1} */
        /* RMS 去归一化 q: dQ(对 q')→dQ(对原始 q): dq_i=dq'_i/r - q_i·(dq'·q)/(D·r³) */
        { double dot=0; for(int i=0;i<D;i++) dot+=(double)dQ[t*D+i]*Q[t*D+i];
          double r=RQ[t], r3=r*r*r;
          for(int i=0;i<D;i++){ double dq=dQ[t*D+i];
              dQ[t*D+i]=(float)(dq/r - Q[t*D+i]*dot/((double)D*r3)); } }
        /* RMS 去归一化 k: 先乘 elu'(k') 得对 k' 梯度,再同样去归一化 */
        { for(int i=0;i<D;i++) dK[t*D+i]*=elu_p(K[t*D+i]);
          double dot=0; for(int i=0;i<D;i++) dot+=(double)dK[t*D+i]*K[t*D+i];
          double r=RK[t], r3=r*r*r;
          for(int i=0;i<D;i++){ double dk=dK[t*D+i];
              dK[t*D+i]=(float)(dk/r - K[t*D+i]*dot/((double)D*r3)); } }
    }
    /* 投影梯度(dK/dQ 已含 elu链式与RMS去归一化,直接投影) */
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        for(int e=0;e<D;e++){
            dWk[d*D+e]+=dK[t*D+d]*X[t*D+e];
            dWq[d*D+e]+=dQ[t*D+d]*X[t*D+e];
            dWv[d*D+e]+=dV[t*D+d]*X[t*D+e];
        }
    }
    /* embedding 反传(dK/dQ 已处理链式与归一化) */
    for(int t=0;t<L;t++) for(int e=0;e<D;e++){ double s=0;
        for(int d=0;d<D;d++){
            s+=(double)dQ[t*D+d]*Wq[d*D+e] + (double)dK[t*D+d]*Wk[d*D+e] + (double)dV[t*D+d]*Wv[d*D+e];
        }
        dE[S[t]*D+e]+=(float)s; }

    /* 梯度裁剪 + adamw */
    { double gn2=0;
      for(int i=0;i<V*D;i++) gn2+=(double)dE[i]*dE[i];
      for(int i=0;i<D*D;i++) gn2+=(double)dWq[i]*dWq[i]+(double)dWk[i]*dWk[i]+(double)dWv[i]*dWv[i];
      for(int i=0;i<V*D;i++) gn2+=(double)dWo[i]*dWo[i];
      double gn=sqrt(gn2); float maxg=0.5f; float sc=(gn>maxg)?(float)(maxg/gn):1.0f;
      for(int i=0;i<V*D;i++){ dE[i]*=sc; dWo[i]*=sc; }
      for(int i=0;i<D*D;i++){ dWq[i]*=sc; dWk[i]*=sc; dWv[i]*=sc; } }
    float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
    #define ADAMW(P,M,Vv,G,N) do{ for(int i=0;i<(N);i++){ M[i]=b1*M[i]+(1-b1)*G[i]; Vv[i]=b2*Vv[i]+(1-b2)*G[i]*G[i]; P[i]-=lr*(M[i]/bc1)/(sqrtf(Vv[i]/bc2)+eps_a);} }while(0)
    ADAMW(E,mE,vE,dE,V*D);
    ADAMW(Wq,mQ,vQ,dWq,D*D);
    ADAMW(Wk,mK,vK,dWk,D*D);
    ADAMW(Wv,mV,vV,dWv,D*D);
    ADAMW(Wo,mO,vO,dWo,V*D);
    #undef ADAMW
    return loss/(L-1);
}

/* 采样验证(眼见为实): 训练后自回归生成,看模型自己能否复现周期规律 */
static void sample(void){
    forward();
    int correct=0;
    for(int t=0;t<L-1;t++){ int pred=0; float mx=LG[t*V];
        for(int j=1;j<V;j++) if(LG[t*V+j]>mx){mx=LG[t*V+j];pred=j;}
        if(pred==S[t+1]) correct++; }
    printf("  训练序列逐位 argmax 命中 %d/%d", correct, L-1);
    printf(" | 自回归生成: ");
    int buf[16]; for(int i=0;i<16;i++) buf[i]=S[i%4];
    int expect[4]={9,3,7,1};
    int ok=0;
    for(int step=0;step<8;step++){
        for(int i=0;i<L;i++) S[i]=buf[i];
        forward();
        int pred=0; float mx=LG[(L-2)*V]; for(int j=1;j<V;j++) if(LG[(L-2)*V+j]>mx){mx=LG[(L-2)*V+j];pred=j;}
        printf("%d ",pred);
        if(pred==expect[step%4]) ok++;
        for(int i=0;i<L-1;i++) buf[i]=buf[i+1];
        buf[L-1]=pred;
    }
    printf("(期望 9 3 7 1 循环, 命中 %d/8)\n", ok);
    /* 还原训练序列 */
    gen_seq();
}

int main(void){
    gen_seq();
    pcg_seed(7);
    for(int i=0;i<V*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; }
    for(int i=0;i<V*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    printf("== P2 因果线性注意力语言模型 (V=%d d=%d L=%d) ==\n", V, D, L);
    printf("因果:只看过去,state增量,内存O(d²); 规律:周期4 motif [3 7 1 9]\n");
    float lr=0.003f,b1=0.9f,b2=0.999f,eps_a=1e-8f;
    for(int ep=1;ep<=EPOCHS;ep++){
        double loss=train_epoch(lr,b1,b2,eps_a,ep);
        if(ep==1||ep%20==0||ep==EPOCHS) printf("  epoch %3d  loss=%.6f\n",ep,loss);
    }
    printf("== 采样验证 ==\n");
    sample();
    return 0;
}
