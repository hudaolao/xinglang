// 星算语 fromzero · P2 核心:单头线性注意力语言模型(六智者基座架构)
// 线性注意力(非softmax): y[t] = Σ_s (q_t·k_s) v_s / (Σ_s q_t·k_s + eps)
//   复杂度 O(T·d²) 而非 O(T²),无 softmax 指数,可因果 state 增量
// 确定性:合同PCG32 + 固定归约序 + 解析反向 + adamw
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 32        /* 词表 */
#define D 32        /* d_model */
#define L 16        /* 序列长 */
#define EPOCHS 200

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

/* 数据: seq[t]=(t+offset) mod V —— 全局偏移规律 */
static int S[L];
static void gen_seq(void){
    int motif[4]={3,7,1,9};   /* 周期4的确定性语言模式 */
    for(int t=0;t<L;t++) S[t]=motif[t%4];
}

/* 参数 + adamw 状态(全局) */
static float E[V*D], Wq[D*D], Wk[D*D], Wv[D*D], Wo[V*D];
static float mE[V*D],mQ[D*D],mK[D*D],mV[D*D],mO[V*D];
static float vE[V*D],vQ[D*D],vK[D*D],vV[D*D],vO[V*D];

/* 前向中间量 */
static float X[L*D], Q[L*D], K[L*D], VV[L*D], Y[L*D], LG[L*V], A[L*L];
static float SATT[L*L];   /* q·k 原始值,反向求 elu' 用 */

static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
/* 非负核: elu(x)+1 > 0 恒成立,分母 z 稳定不爆炸 */
static float elu(float x){ return x>0 ? x : expm1f(x); }
static float elu_p(float x){ return x>0 ? 1.0f : expf(x); }

static void forward(void){
    float eps=1e-6f;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) X[t*D+d]=E[S[t]*D+d];
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        double sq=0,sk=0,sv=0;
        for(int e=0;e<D;e++){ sq+=(double)Wq[d*D+e]*X[t*D+e]; sk+=(double)Wk[d*D+e]*X[t*D+e]; sv+=(double)Wv[d*D+e]*X[t*D+e]; }
        Q[t*D+d]=(float)sq; K[t*D+d]=(float)sk; VV[t*D+d]=(float)sv;
    }
    for(int t=0;t<L;t++){
        float z=0; float a[D]; for(int d=0;d<D;d++) a[d]=0;
        for(int s=0;s<L;s++){
            float sval=0; for(int d=0;d<D;d++) sval+=Q[t*D+d]*K[s*D+d];
            SATT[t*L+s]=sval;
            float w=elu(sval)+1.0f;   /* 非负核,保证 z>0 稳定 */
            A[t*L+s]=w; z+=w; for(int d=0;d<D;d++) a[d]+=w*VV[s*D+d];
        }
        z+=eps;
        for(int d=0;d<D;d++) Y[t*D+d]=a[d]/z;
    }
    for(int t=0;t<L;t++) for(int j=0;j<V;j++){
        double s=0; for(int d=0;d<D;d++) s+=(double)Wo[j*D+d]*Y[t*D+d];
        LG[t*V+j]=(float)s;
    }
}

static double train_epoch(float lr, float b1, float b2, float eps_a, int ep){
    float eps=1e-6f;
    forward();
    /* softmax 反向 + loss */
    static float dLG[L*V];
    double loss=0;
    for(int t=0;t<L-1;t++){ float p[V]; softmax_row(&LG[t*V],p,V);
        int target=S[t+1];   /* 预测下一个token(自回归) */
        double q=(double)p[target]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++) dLG[t*V+j]=(j==target)?(p[j]-1.0f):p[j]; }
    /* t=L-1 无下一个, dLG 保持 0,不参与 loss */

    /* dY, dWo */
    static float dY[L*D], dWo[V*D], dWq[D*D], dWk[D*D], dWv[D*D], dE[V*D];
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){ double s=0;
        for(int j=0;j<V;j++) s+=(double)dLG[t*V+j]*Wo[j*D+d]; dY[t*D+d]=(float)s; }
    for(int j=0;j<V;j++) for(int d=0;d<D;d++){ double s=0;
        for(int t=0;t<L;t++) s+=(double)dLG[t*V+j]*Y[t*D+d]; dWo[j*D+d]=(float)s; }

    /* 线性注意力反向 */
    static float dQ[L*D], dK[L*D], dV[L*D];
    for(int t=0;t<L;t++){ float z=0; for(int s=0;s<L;s++) z+=A[t*L+s]; z+=eps;
        float dz=0; for(int d=0;d<D;d++) dz+=dY[t*D+d]*Y[t*D+d]; dz=-dz/z;
        for(int s=0;s<L;s++){
            float w=A[t*L+s]; float dw=dz;
            for(int d=0;d<D;d++) dw+=(dY[t*D+d]/z)*VV[s*D+d];
            float dws=dw*elu_p(SATT[t*L+s]);   /* elu 核链式 */
            for(int d=0;d<D;d++){
                float da=dY[t*D+d]/z;
                dV[s*D+d]+=w*da;
                dQ[t*D+d]+=dws*K[s*D+d];
                dK[s*D+d]+=dws*Q[t*D+d];
            }
        }
    }
    /* 投影梯度 + embedding 反传 */
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) for(int e=0;e<D;e++){
        dWq[d*D+e]+=dQ[t*D+d]*X[t*D+e];
        dWk[d*D+e]+=dK[t*D+d]*X[t*D+e];
        dWv[d*D+e]+=dV[t*D+d]*X[t*D+e];
    }
    for(int t=0;t<L;t++) for(int e=0;e<D;e++){ double s=0;
        for(int d=0;d<D;d++) s+=(double)dQ[t*D+d]*Wq[d*D+e] + (double)dK[t*D+d]*Wk[d*D+e] + (double)dV[t*D+d]*Wv[d*D+e];
        dE[S[t]*D+e]+=(float)s; }

    /* 梯度裁剪(线性注意力梯度易爆,全局L2范数裁剪) */
    { double gn2=0;
      for(int i=0;i<V*D;i++) gn2+=(double)dE[i]*dE[i];
      for(int i=0;i<D*D;i++) gn2+=(double)dWq[i]*dWq[i]+(double)dWk[i]*dWk[i]+(double)dWv[i]*dWv[i];
      for(int i=0;i<V*D;i++) gn2+=(double)dWo[i]*dWo[i];
      double gn=sqrt(gn2);
      float maxg=1.0f; float sc=(gn>maxg)?(float)(maxg/gn):1.0f;
      for(int i=0;i<V*D;i++){ dE[i]*=sc; dWo[i]*=sc; }
      for(int i=0;i<D*D;i++){ dWq[i]*=sc; dWk[i]*=sc; dWv[i]*=sc; }
    }
    /* adamw 更新 */
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

int main(void){
    gen_seq();
    pcg_seed(7);
    for(int i=0;i<V*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; }
    for(int i=0;i<V*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;

    printf("== P2 单头线性注意力语言模型 (V=%d d=%d L=%d) ==\n", V, D, L);
    printf("数据规律: 周期4 motif [3 7 1 9] 循环 —— 靠检索历史结构预测下一个\n");
    float lr=0.003f, b1=0.9f, b2=0.999f, eps_a=1e-8f;
    for(int ep=1; ep<=EPOCHS; ep++){
        double loss=train_epoch(lr,b1,b2,eps_a,ep);
        if(ep==1||ep%10==0||ep==EPOCHS) printf("  epoch %3d  loss=%.6f\n", ep, loss);
    }
    return 0;
}
