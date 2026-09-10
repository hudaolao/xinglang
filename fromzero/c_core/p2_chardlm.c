// P2 字符级中文语言模型: 因果线性注意力(带全部稳定修复) 训练真实文本
// 数据: 两首唐诗(静夜思+春晓), 字符级
// 目标: loss 显著下降 + 自回归生成像样的中文文本(眼见为实)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define D 32
#define LMAX 256
#define MAXV 80
#define EPOCHS 400

static const char* TEXT =
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。";

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static int V;          /* 实际字符数 */
static int L;          /* 实际序列长度(文本长度-1, 预测到末字符) */
static int S[LMAX];    /* 文本字符 id */
static char CHARS[MAXV];
static void load_text(void){
    int n=(int)strlen(TEXT);
    V=0;
    for(int i=0;i<n;i++){
        int found=0;
        for(int j=0;j<V;j++) if(CHARS[j]==TEXT[i]){ found=1; break; }
        if(!found) CHARS[V++]=TEXT[i];
    }
    for(int i=0;i<n;i++)
        for(int j=0;j<V;j++) if(CHARS[j]==TEXT[i]){ S[i]=j; break; }
    L=n-1;   /* 训练 t=0..L-1 预测 S[t+1] 到 S[L]=末字符 */
}

static float E[MAXV*D], Wq[D*D], Wk[D*D], Wv[D*D], Wo[MAXV*D];
static float mE[MAXV*D],mQ[D*D],mK[D*D],mV[D*D],mO[MAXV*D];
static float vE[MAXV*D],vQ[D*D],vK[D*D],vV[D*D],vO[MAXV*D];

static float X[LMAX*D], Q[LMAX*D], K_[LMAX*D], VV[LMAX*D], Y[LMAX*D], LG[LMAX*MAXV];
static float STATE[LMAX*D*D], Z[LMAX*D], NUM[LMAX*D], DEN[LMAX];
static float RQ[LMAX], RK[LMAX];
static float SDEN[LMAX];

static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
static float elu(float x){ return x>0 ? x : expm1f(x); }
static float elu_p(float x){ return x>0 ? 1.0f : expf(x); }

static void forward(void){
    float eps=1e-3f;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) X[t*D+d]=E[S[t]*D+d];
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        double sq=0,sk=0,sv=0;
        for(int e=0;e<D;e++){ sq+=(double)Wq[d*D+e]*X[t*D+e]; sk+=(double)Wk[d*D+e]*X[t*D+e]; sv+=(double)Wv[d*D+e]*X[t*D+e]; }
        Q[t*D+d]=(float)sq; K_[t*D+d]=(float)sk; VV[t*D+d]=(float)sv;
    }
    for(int t=0;t<L;t++){
        double q2=0,k2=0;
        for(int d=0;d<D;d++){ q2+=(double)Q[t*D+d]*Q[t*D+d]; k2+=(double)K_[t*D+d]*K_[t*D+d]; }
        double rq=sqrt(q2/D)+1e-6, rk=sqrt(k2/D)+1e-6;
        RQ[t]=(float)rq; RK[t]=(float)rk;
        for(int d=0;d<D;d++){ Q[t*D+d]=(float)((double)Q[t*D+d]/rq); K_[t*D+d]=(float)((double)K_[t*D+d]/rk); }
    }
    float st[D*D]; for(int i=0;i<D*D;i++) st[i]=0;
    float z[D]; for(int i=0;i<D;i++) z[i]=0;
    for(int t=0;t<L;t++){
        for(int i=0;i<D;i++){ float ki=elu(K_[t*D+i])+1.0f;
            for(int j=0;j<D;j++) st[i*D+j]+=ki*VV[t*D+j];
            z[i]+=ki; }
        for(int i=0;i<D*D;i++) STATE[t*D*D+i]=st[i];
        for(int i=0;i<D;i++) Z[t*D+i]=z[i];
        for(int j=0;j<D;j++){ double s=0; for(int i=0;i<D;i++) s+=(double)st[i*D+j]*Q[t*D+i]; NUM[t*D+j]=(float)s; }
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
    forward();
    static float dLG[LMAX*MAXV];
    double loss=0;
    for(int t=0;t<L;t++){ float p[MAXV]; softmax_row(&LG[t*V],p,V);
        int target=S[t+1]; double q=(double)p[target]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++) dLG[t*V+j]=(j==target)?(p[j]-1.0f):p[j]; }

    static float dY[LMAX*D], dWo[MAXV*D], dWq[D*D], dWk[D*D], dWv[D*D], dE[MAXV*D];
    static float dQ[LMAX*D], dK[LMAX*D], dV[LMAX*D];
    for(int i=0;i<D*D;i++){ dWq[i]=0; dWk[i]=0; dWv[i]=0; }
    for(int i=0;i<MAXV*D;i++) dE[i]=0;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){ double s=0;
        for(int j=0;j<V;j++) s+=(double)dLG[t*V+j]*Wo[j*D+d]; dY[t*D+d]=(float)s; }
    for(int j=0;j<V;j++) for(int d=0;d<D;d++){ double s=0;
        for(int t=0;t<L;t++) s+=(double)dLG[t*V+j]*Y[t*D+d]; dWo[j*D+d]=(float)s; }

    static float G[D*D], Gz[D], dstate[D*D];
    for(int i=0;i<D*D;i++) G[i]=0;
    for(int i=0;i<D;i++) Gz[i]=0;
    for(int t=L-1;t>=0;t--){
        float dnum[D]; for(int j=0;j<D;j++) dnum[j]=dY[t*D+j]/DEN[t];
        double dden=0; for(int j=0;j<D;j++) dden+=(double)dY[t*D+j]*NUM[t*D+j]; dden=-dden/((double)DEN[t]*DEN[t]);
        dden*=(double)SDEN[t];
        for(int i=0;i<D;i++){
            double dqi=(double)((float)(dden*Z[t*D+i]));
            for(int j=0;j<D;j++) dqi+=(double)dnum[j]*STATE[t*D*D+i*D+j];
            dQ[t*D+i]=(float)dqi;
        }
        for(int i=0;i<D;i++) Gz[i]+=(float)(dden*Q[t*D+i]);
        for(int i=0;i<D;i++) for(int j=0;j<D;j++)
            dstate[i*D+j]=G[i*D+j]+dnum[j]*Q[t*D+i];
        for(int i=0;i<D;i++){ double dk=Gz[i]; for(int j=0;j<D;j++) dk+=(double)dstate[i*D+j]*VV[t*D+j]; dK[t*D+i]=(float)dk; }
        for(int j=0;j<D;j++){ double dv=0; for(int i=0;i<D;i++) dv+=(double)dstate[i*D+j]*(elu(K_[t*D+i])+1.0f); dV[t*D+j]=(float)dv; }
        for(int i=0;i<D*D;i++) G[i]=dstate[i];
        { double dot=0; for(int i=0;i<D;i++) dot+=(double)dQ[t*D+i]*Q[t*D+i];
          double r=RQ[t], r3=r*r*r;
          for(int i=0;i<D;i++){ double dq=dQ[t*D+i];
              dQ[t*D+i]=(float)(dq/r - Q[t*D+i]*dot/((double)D*r3)); } }
        { for(int i=0;i<D;i++) dK[t*D+i]*=elu_p(K_[t*D+i]);
          double dot=0; for(int i=0;i<D;i++) dot+=(double)dK[t*D+i]*K_[t*D+i];
          double r=RK[t], r3=r*r*r;
          for(int i=0;i<D;i++){ double dk=dK[t*D+i];
              dK[t*D+i]=(float)(dk/r - K_[t*D+i]*dot/((double)D*r3)); } }
    }
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        for(int e=0;e<D;e++){
            dWk[d*D+e]+=dK[t*D+d]*X[t*D+e];
            dWq[d*D+e]+=dQ[t*D+d]*X[t*D+e];
            dWv[d*D+e]+=dV[t*D+d]*X[t*D+e];
        }
    }
    for(int t=0;t<L;t++) for(int e=0;e<D;e++){ double s=0;
        for(int d=0;d<D;d++){
            s+=(double)dQ[t*D+d]*Wq[d*D+e] + (double)dK[t*D+d]*Wk[d*D+e] + (double)dV[t*D+d]*Wv[d*D+e];
        }
        dE[S[t]*D+e]+=(float)s; }

    { double gn2=0;
      for(int i=0;i<MAXV*D;i++) gn2+=(double)dE[i]*dE[i];
      for(int i=0;i<D*D;i++) gn2+=(double)dWq[i]*dWq[i]+(double)dWk[i]*dWk[i]+(double)dWv[i]*dWv[i];
      for(int i=0;i<MAXV*D;i++) gn2+=(double)dWo[i]*dWo[i];
      double gn=sqrt(gn2); float maxg=0.5f; float sc=(gn>maxg)?(float)(maxg/gn):1.0f;
      for(int i=0;i<MAXV*D;i++){ dE[i]*=sc; dWo[i]*=sc; }
      for(int i=0;i<D*D;i++){ dWq[i]*=sc; dWk[i]*=sc; dWv[i]*=sc; } }
    float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
    #define ADAMW(P,M,Vv,G,N) do{ for(int i=0;i<(N);i++){ M[i]=b1*M[i]+(1-b1)*G[i]; Vv[i]=b2*Vv[i]+(1-b2)*G[i]*G[i]; P[i]-=lr*(M[i]/bc1)/(sqrtf(Vv[i]/bc2)+eps_a);} }while(0)
    ADAMW(E,mE,vE,dE,MAXV*D);
    ADAMW(Wq,mQ,vQ,dWq,D*D);
    ADAMW(Wk,mK,vK,dWk,D*D);
    ADAMW(Wv,mV,vV,dWv,D*D);
    ADAMW(Wo,mO,vO,dWo,MAXV*D);
    #undef ADAMW
    return loss/L;
}

static void init(void){
    pcg_seed(20260910);
    for(int i=0;i<MAXV*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; }
    for(int i=0;i<MAXV*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<MAXV*D;i++){ mE[i]=vE[i]=0; mO[i]=vO[i]=0; }
    for(int i=0;i<D*D;i++){ mQ[i]=vQ[i]=0; mK[i]=vK[i]=0; mV[i]=vV[i]=0; }
}

/* 自回归生成: seed=文本前 L 个字符, 滚动生成 60 个字符 */
static void generate(void){
    printf("  生成: ");
    int buf[LMAX]; for(int i=0;i<L;i++) buf[i]=S[i];
    char out[LMAX*2]; int on=0;
    for(int i=0;i<L;i++) out[on++]=CHARS[buf[i]];
    for(int step=0;step<60;step++){
        for(int i=0;i<L;i++) S[i]=buf[i];
        forward();
        int pred=0; float mx=LG[(L-1)*V]; for(int j=1;j<V;j++) if(LG[(L-1)*V+j]>mx){mx=LG[(L-1)*V+j];pred=j;}
        out[on++]=CHARS[pred];
        for(int i=0;i<L-1;i++) buf[i]=buf[i+1];
        buf[L-1]=pred;
    }
    out[on]=0;
    printf("%s\n", out);
    load_text();  /* 还原 S */
}

int main(void){
    load_text();
    printf("== P2 字符级中文语言模型 (V=%d, L=%d, d=%d) ==\n", V, L, D);
    printf("文本: %s\n", TEXT);
    printf("字符集: ");
    for(int j=0;j<V;j++) printf("%c", CHARS[j]);
    printf("\n\n");
    init();
    float lr=0.003f,b1=0.9f,b2=0.999f,eps_a=1e-8f;
    for(int ep=1;ep<=EPOCHS;ep++){
        double loss=train_epoch(lr,b1,b2,eps_a,ep);
        if(ep==1||ep%50==0||ep==EPOCHS) printf("  epoch %3d  loss=%.4f\n",ep,loss);
    }
    printf("\n== 采样验证 ==\n");
    generate();
    return 0;
}
