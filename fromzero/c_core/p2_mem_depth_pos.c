// 记忆深度实验: 因果线性注意力 + 可学习绝对位置编码
// 对照无位置编码版(p2_mem_depth.c): 看 PE 能否让 query 区分"几步前", 突破记忆深度
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 32
#define D 32
#define L 24
#define EPOCHS 300

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static int K;           /* 周期 */
static int S[L+1];
static void gen_seq(void){
    uint32_t r=pcg_next();
    for(int i=0;i<K;i++){ S[i]=(int)(r%V); r=pcg_next(); }
    for(int t=K;t<L+1;t++) S[t]=S[t-K];
}

static float E[V*D], Wq[D*D], Wk[D*D], Wv[D*D], Wo[V*D];
static float PE[L*D];   /* 可学习绝对位置编码 */
static float mE[V*D],mQ[D*D],mK[D*D],mV[D*D],mO[V*D],mP[L*D];
static float vE[V*D],vQ[D*D],vK[D*D],vV[D*D],vO[V*D],vP[L*D];

static float X[L*D], Q[L*D], K_[L*D], VV[L*D], Y[L*D], LG[L*V];
static float STATE[L*D*D], Z[L*D], NUM[L*D], DEN[L];
static float RQ[L], RK[L];
static float SDEN[L];

static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
static float elu(float x){ return x>0 ? x : expm1f(x); }
static float elu_p(float x){ return x>0 ? 1.0f : expf(x); }

static void forward(void){
    float eps=1e-3f;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) X[t*D+d]=E[S[t]*D+d]+PE[t*D+d];
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
    static float dLG[L*V];
    double loss=0;
    for(int t=0;t<L;t++){ float p[V]; softmax_row(&LG[t*V],p,V);
        int target=S[t+1]; double q=(double)p[target]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++) dLG[t*V+j]=(j==target)?(p[j]-1.0f):p[j]; }

    static float dY[L*D], dWo[V*D], dWq[D*D], dWk[D*D], dWv[D*D], dE[V*D];
    static float dPE[L*D];
    static float dQ[L*D], dK[L*D], dV[L*D];
    for(int i=0;i<D*D;i++){ dWq[i]=0; dWk[i]=0; dWv[i]=0; }
    for(int i=0;i<V*D;i++) dE[i]=0;
    for(int i=0;i<L*D;i++) dPE[i]=0;
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
        dE[S[t]*D+e]+=(float)s;
        dPE[t*D+e]+=(float)s; }   /* 位置嵌入梯度: X=E[S]+PE, dPE=dX */

    { double gn2=0;
      for(int i=0;i<V*D;i++) gn2+=(double)dE[i]*dE[i];
      for(int i=0;i<D*D;i++) gn2+=(double)dWq[i]*dWq[i]+(double)dWk[i]*dWk[i]+(double)dWv[i]*dWv[i];
      for(int i=0;i<V*D;i++) gn2+=(double)dWo[i]*dWo[i];
      for(int i=0;i<L*D;i++) gn2+=(double)dPE[i]*dPE[i];
      double gn=sqrt(gn2); float maxg=0.5f; float sc=(gn>maxg)?(float)(maxg/gn):1.0f;
      for(int i=0;i<V*D;i++){ dE[i]*=sc; dWo[i]*=sc; }
      for(int i=0;i<D*D;i++){ dWq[i]*=sc; dWk[i]*=sc; dWv[i]*=sc; }
      for(int i=0;i<L*D;i++) dPE[i]*=sc; }
    float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
    #define ADAMW(P,M,Vv,G,N) do{ for(int i=0;i<(N);i++){ M[i]=b1*M[i]+(1-b1)*G[i]; Vv[i]=b2*Vv[i]+(1-b2)*G[i]*G[i]; P[i]-=lr*(M[i]/bc1)/(sqrtf(Vv[i]/bc2)+eps_a);} }while(0)
    ADAMW(E,mE,vE,dE,V*D);
    ADAMW(Wq,mQ,vQ,dWq,D*D);
    ADAMW(Wk,mK,vK,dWk,D*D);
    ADAMW(Wv,mV,vV,dWv,D*D);
    ADAMW(Wo,mO,vO,dWo,V*D);
    ADAMW(PE,mP,vP,dPE,L*D);
    #undef ADAMW
    return loss/L;
}

static void init(void){
    gen_seq();
    for(int i=0;i<V*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; }
    for(int i=0;i<V*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<L*D;i++) PE[i]=(float)(pcg_next()>>8)/16777216.0f*0.05f;
    for(int i=0;i<V*D;i++){ mE[i]=vE[i]=0; mO[i]=vO[i]=0; }
    for(int i=0;i<D*D;i++){ mQ[i]=vQ[i]=0; mK[i]=vK[i]=0; mV[i]=vV[i]=0; }
    for(int i=0;i<L*D;i++){ mP[i]=vP[i]=0; }
}

static int hit_train(void){
    forward();
    int c=0;
    for(int t=0;t<L;t++){ int pred=0; float mx=LG[t*V];
        for(int j=1;j<V;j++) if(LG[t*V+j]>mx){mx=LG[t*V+j];pred=j;}
        if(pred==S[t+1]) c++; }
    return c;
}
static int auto_reg(void){
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
}

int main(void){
    printf("== 记忆深度实验: 因果线性注意力 + 可学习位置编码(PE) ==\n");
    printf("任务: 周期K随机模式, 位置t预测S[t+1]=S[t+1-K]\n\n");
    for(int k=1;k<=6;k++){
        K=k;
        pcg_seed(1000+k);  /* 每个k固定种子,可比 */
        init();
        float lr=0.003f,b1=0.9f,b2=0.999f,eps_a=1e-8f;
        double best=1e9; int best_ep=0;
        for(int ep=1;ep<=EPOCHS;ep++){
            double loss=train_epoch(lr,b1,b2,eps_a,ep);
            if(loss<best){ best=loss; best_ep=ep; }
        }
        int h=hit_train();
        int ar=auto_reg();
        printf("K=%d  best loss=%.4f (ep%d)  训练命中 %d/%d  自回归 %d/10%s\n",
            k, best, best_ep, h, L, ar, (ar>=8)?"  ✓ 学会周期":"");
    }
    return 0;
}
