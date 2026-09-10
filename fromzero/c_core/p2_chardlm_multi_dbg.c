// P2 字符级中文语言模型 v2: 多诗语料 + 滑动窗口多样本训练
// 目标: 多样本让基座从"背诵单序列"走向"学到诗的统计分布"(泛化)
// 数据: 10 首五言绝句拼接, 每 epoch 随机窗口采样训练
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define D 64
#define LMAX 300
#define MAXV 160
#define EPOCHS 400
#define WIN 48      /* 滑动窗口长度 */
#define NSAMP 24    /* 每 epoch 训练样本数 */

static const char* POEMS[10]={
 "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
 "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
 "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
 "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
 "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
 "鹅鹅鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
 "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
 "松下问童子，言师采药去。只在此山中，云深不知处。",
 "空山不见人，但闻人语响。返景入深林，复照青苔上。",
 "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。",
};

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static int V;                     /* 全局唯一字符数 */
static char CHARS[MAXV];
static int ALLID[LMAX*3];         /* 拼接语料字符 id */
static int N;                     /* 语料长度 */
static float lr=0.003f,b1=0.9f,b2=0.999f,eps_a=1e-8f;
static void load_corpus(void){
    char buf[LMAX*4]; int pos=0;
    for(int p=0;p<10;p++){ int n=(int)strlen(POEMS[p]); for(int i=0;i<n;i++) buf[pos++]=POEMS[p][i]; }
    V=0;
    for(int i=0;i<pos;i++){ int found=0; for(int j=0;j<V;j++) if(CHARS[j]==buf[i]){found=1;break;} if(!found) CHARS[V++]=buf[i]; }
    for(int i=0;i<pos;i++) for(int j=0;j<V;j++) if(CHARS[j]==buf[i]){ ALLID[i]=j; break; }
    N=pos;
}

static float E[MAXV*D], Wq[D*D], Wk[D*D], Wv[D*D], Wo[MAXV*D];
static float mE[MAXV*D],mQ[D*D],mK[D*D],mV[D*D],mO[MAXV*D];
static float vE[MAXV*D],vQ[D*D],vK[D*D],vV[D*D],vO[MAXV*D];

static float X[LMAX*D], Q[LMAX*D], K_[LMAX*D], VV[LMAX*D], Y[LMAX*D], LG[LMAX*MAXV];
static float STATE[LMAX*D*D], Z[LMAX*D], NUM[LMAX*D], DEN[LMAX];
static float RQ[LMAX], RK[LMAX];
static float SDEN[LMAX];
static int S[LMAX], L;

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

static double train_window(int start){
    for(int i=0;i<WIN;i++) S[i]=ALLID[start+i];
    L=WIN;
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
    float bc1=1.0f, bc2=1.0f;   /* 省略 bias correction, 早期动量略大但 lr 小, 稳定 */
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
    for(int i=0;i<MAXV*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f; }
    for(int i=0;i<MAXV*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f;
    for(int i=0;i<MAXV*D;i++){ mE[i]=vE[i]=0; mO[i]=vO[i]=0; }
    for(int i=0;i<D*D;i++){ mQ[i]=vQ[i]=0; mK[i]=vK[i]=0; mV[i]=vV[i]=0; }
}

static void generate(void){
    /* seed = 第1首诗前 WIN 个字符, 自回归生成 60 字符 */
    printf("  生成(seed=静夜思): ");
    int seedn=(int)strlen(POEMS[0]);
    int buf[LMAX]; for(int i=0;i<WIN;i++) buf[i]=(i<seedn)?ALLID[i]:ALLID[(i-seedn)];  /* seed: 静夜思+续接 */
    for(int i=0;i<WIN;i++) S[i]=buf[i];
    L=WIN;
    char out[LMAX*2]; int on=0;
    for(int i=0;i<seedn;i++) out[on++]=CHARS[buf[i]];
    for(int step=0;step<60;step++){
        forward();
        int pred=0; float mx=LG[(L-1)*V]; for(int j=1;j<V;j++) if(LG[(L-1)*V+j]>mx){mx=LG[(L-1)*V+j];pred=j;}
        out[on++]=CHARS[pred];
        for(int i=0;i<L-1;i++) buf[i]=buf[i+1];
        buf[L-1]=pred;
        for(int i=0;i<L;i++) S[i]=buf[i];
    }
    out[on]=0;
    printf("%s\n", out);
}

int main(void){
    load_corpus();
    printf("== P2 中文语言模型 v2: 10首唐诗, V=%d, N=%d, d=%d, 窗口%d x %d样本/epoch ==\n", V, N, D, WIN, NSAMP);
    init(); printf("[dbg] init done\\n"); fflush(stdout);
    for(int ep=1;ep<=EPOCHS;ep++){
        double tot=0;
        for(int s=0;s<NSAMP;s++){
            int start=(int)(((uint32_t)pcg_next()>>8) % (N-WIN-1));
            tot+=train_window(start);
        }
        if(ep==1||ep%50==0||ep==EPOCHS) printf("  epoch %3d  loss=%.4f\n",ep,tot/NSAMP);
    }
    printf("\n== 采样验证(训练后生成) ==\n");
    generate();
    return 0;
}

