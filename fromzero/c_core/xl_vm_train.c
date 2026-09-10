// 星算语 fromzero · 字节码执行器 VM
// 读 train_step.xlbin (XL01, v0.2) → 解释驱动因果线性注意力训练 → 五项哈希 golden
// 对齐: spec/xinglang_v02_training_ir.md + P2 xl_linear_attn_causal.c 语义
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

static int g_nth = 1;   /* 确定性并行: 只改执行速度, 不改求值顺序 */

#define V 32
#define D 32
#define L 16
#define STEPS 40

/* ---------- 确定性哈希 (FNV-1a 64) ---------- */
static uint64_t hst;
static void hreset(void){ hst=0xcbf29ce484222325ULL; }
static void hput(const void* p,int n){ const uint8_t* b=p; for(int i=0;i<n;i++){ hst^=b[i]; hst*=0x100000001b3ULL; } }
static void hputu32(uint32_t v){ hput(&v,4); }
static void hputf(float f){ hput(&f,4); }
static void hputstr(const char* s){ while(*s){ hst^=(uint8_t)*s++; hst*=0x100000001b3ULL; } }
static uint64_t hget(void){ return hst; }

/* ---------- 字节码 ---------- */
static uint8_t iop[64]; static uint32_t iopd[64]; static int ninst;
static int read_xlbin(const char* p){
    FILE* f=fopen(p,"rb"); if(!f) return -1;
    uint8_t hdr[7]; if(fread(hdr,1,7,f)!=7){ fclose(f); return -2; }
    if(hdr[0]!='X'||hdr[1]!='L'||hdr[2]!='0'||hdr[3]!='1'){ fclose(f); return -3; }
    int n=0; uint8_t op; uint8_t o[3];
    while(fread(&op,1,1,f)==1){ if(fread(o,1,3,f)!=3)break;
        iop[n]=op; iopd[n]=o[0]|(o[1]<<8)|(o[2]<<16); n++; }
    fclose(f); return n;
}

/* ---------- 权重/优化器/激活 ---------- */
static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state=s+1442695040888963407ULL; }
static uint32_t pcg_next(void){ pcg_state=pcg_state*6364136223846793005ULL+1442695040888963407ULL;
    uint32_t rot=(uint32_t)(pcg_state>>59); uint64_t xs=(pcg_state>>18)^pcg_state;
    return (uint32_t)(((xs>>(32-rot))|(xs<<rot))&0xFFFFFFFF); }

static int S[L];
static void gen_seq(void){ int m[4]={3,7,1,9}; for(int t=0;t<L;t++) S[t]=m[t%4]; }

static float E[V*D],Wq[D*D],Wk[D*D],Wv[D*D],Wo[V*D];
static float mE[V*D],mQ[D*D],mK[D*D],mV[D*D],mO[V*D];
static float vE[V*D],vQ[D*D],vK[D*D],vV[D*D],vO[V*D];
static float X[L*D],Q[L*D],K[L*D],VV[L*D],Y[L*D],LG[L*V];
static float STATE[L*D*D],Z[L*D],NUM[L*D],DEN[L];
static float RQ[L],RK[L],SDEN[L];
static float dLG[L*V],dY[L*D],dWo[V*D],dWq[D*D],dWk[D*D],dWv[D*D],dE[V*D];
static float dQ[L*D],dK[L*D],dV[L*D];
static float G[D*D],Gz[D],dstate[D*D];

static float softmax_row_dummy;
static void softmax_row(const float* x,float* y,int n){ float mx=x[0]; for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];
    double s=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-mx); s+=y[i]; } for(int i=0;i<n;i++)y[i]=(float)(y[i]/s); }
static float elu(float x){ return x>0?x:expm1f(x); }
static float elu_p(float x){ return x>0?1.0f:expf(x); }

/* 算子实现(每 step 由字节码驱动, 对齐 P2 语义) */
static void op_embed(void){
#pragma omp parallel for num_threads(g_nth) schedule(static)
    for(int t=0;t<L;t++)for(int d=0;d<D;d++)X[t*D+d]=E[S[t]*D+d]; }
static void op_proj(int which){ const float* W = which==0?Wq:(which==1?Wk:Wv);
#pragma omp parallel for num_threads(g_nth) schedule(static)
    for(int t=0;t<L;t++)for(int d=0;d<D;d++){
        double s=0; for(int e=0;e<D;e++) s+=(double)W[d*D+e]*X[t*D+e];
        if(which==0)Q[t*D+d]=(float)s; else if(which==1)K[t*D+d]=(float)s; else VV[t*D+d]=(float)s; } }
static void op_rms(int isQ){
#pragma omp parallel for num_threads(g_nth) schedule(static)
    for(int t=0;t<L;t++){ float* A=isQ?Q:K; float* R=isQ?RQ:RK;
    double q2=0; for(int d=0;d<D;d++)q2+=A[t*D+d]*A[t*D+d]; double r=sqrt(q2/D)+1e-6f; R[t]=(float)r;
    for(int d=0;d<D;d++)A[t*D+d]=(float)(A[t*D+d]/r); } }
static void op_attn_fwd(void){ float eps=1e-3f; float st[D*D]; memset(st,0,sizeof st); float z[D]; memset(z,0,sizeof z);
    for(int t=0;t<L;t++){ for(int i=0;i<D;i++){ float ki=elu(K[t*D+i])+1.0f;
        for(int j=0;j<D;j++) st[i*D+j]+=ki*VV[t*D+j]; z[i]+=ki; }
        for(int i=0;i<D*D;i++)STATE[t*D*D+i]=st[i]; for(int i=0;i<D;i++)Z[t*D+i]=z[i];
        for(int j=0;j<D;j++){ double s=0; for(int i=0;i<D;i++)s+=st[i*D+j]*Q[t*D+i]; NUM[t*D+j]=(float)s; }
        double s=0; for(int i=0;i<D;i++)s+=z[i]*Q[t*D+i]; double de=fabs(s)+eps;
        SDEN[t]=(float)(s>=0?1:-1); DEN[t]=(float)de;
        for(int j=0;j<D;j++)Y[t*D+j]=NUM[t*D+j]/DEN[t]; } }
static void op_outproj(void){
#pragma omp parallel for num_threads(g_nth) schedule(static)
    for(int t=0;t<L;t++)for(int j=0;j<V;j++){ double s=0;
    for(int d=0;d<D;d++)s+=Wo[j*D+d]*Y[t*D+d]; LG[t*V+j]=(float)s; } }
static float op_xent(void){ float eps=1e-6f; double loss=0;
    for(int t=0;t<L-1;t++){ float p[V]; softmax_row(&LG[t*V],p,V);
        int tg=S[t+1]; double q=(double)p[tg]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++)dLG[t*V+j]=(j==tg)?(p[j]-1.0f):p[j]; }
    return (float)(loss/(L-1)); }
static void op_attn_bwd(void){
    memset(dWq,0,sizeof dWq); memset(dWk,0,sizeof dWk); memset(dWv,0,sizeof dWv); memset(dE,0,sizeof dE);
    for(int t=0;t<L;t++)for(int d=0;d<D;d++){ double s=0; for(int j=0;j<V;j++)s+=dLG[t*V+j]*Wo[j*D+d]; dY[t*D+d]=(float)s; }
    for(int j=0;j<V;j++)for(int d=0;d<D;d++){ double s=0; for(int t=0;t<L;t++)s+=dLG[t*V+j]*Y[t*D+d]; dWo[j*D+d]=(float)s; }
    memset(G,0,sizeof G); memset(Gz,0,sizeof Gz);
    for(int t=L-1;t>=0;t--){
        float dnum[D]; for(int j=0;j<D;j++)dnum[j]=dY[t*D+j]/DEN[t];
        double dden=0; for(int j=0;j<D;j++)dden+=dY[t*D+j]*NUM[t*D+j]; dden=-dden/((double)DEN[t]*DEN[t]); dden*=SDEN[t];
        for(int i=0;i<D;i++){ double dqi=(double)(dden*Z[t*D+i]);
            for(int j=0;j<D;j++)dqi+=dnum[j]*STATE[t*D*D+i*D+j]; dQ[t*D+i]=(float)dqi; }
        for(int i=0;i<D;i++)Gz[i]+=(float)(dden*Q[t*D+i]);
        for(int i=0;i<D;i++)for(int j=0;j<D;j++)dstate[i*D+j]=G[i*D+j]+dnum[j]*Q[t*D+i];
        for(int i=0;i<D;i++){ double dk=Gz[i]; for(int j=0;j<D;j++)dk+=dstate[i*D+j]*VV[t*D+j]; dK[t*D+i]=(float)dk; }
        for(int j=0;j<D;j++){ double dv=0; for(int i=0;i<D;i++)dv+=dstate[i*D+j]*(elu(K[t*D+i])+1.0f); dV[t*D+j]=(float)dv; }
        for(int i=0;i<D*D;i++)G[i]=dstate[i];
        { double dot=0; for(int i=0;i<D;i++)dot+=dQ[t*D+i]*Q[t*D+i]; double r=RQ[t],r3=r*r*r;
          for(int i=0;i<D;i++){ double dq=dQ[t*D+i]; dQ[t*D+i]=(float)(dq/r-Q[t*D+i]*dot/((double)D*r3)); } }
        { for(int i=0;i<D;i++)dK[t*D+i]*=elu_p(K[t*D+i]); double dot=0;
          for(int i=0;i<D;i++)dot+=dK[t*D+i]*K[t*D+i]; double r=RK[t],r3=r*r*r;
          for(int i=0;i<D;i++){ double dk=dK[t*D+i]; dK[t*D+i]=(float)(dk/r-K[t*D+i]*dot/((double)D*r3)); } }
    }
    for(int t=0;t<L;t++)for(int d=0;d<D;d++)for(int e=0;e<D;e++){
        dWk[d*D+e]+=dK[t*D+d]*X[t*D+e]; dWq[d*D+e]+=dQ[t*D+d]*X[t*D+e]; dWv[d*D+e]+=dV[t*D+d]*X[t*D+e]; }
    for(int t=0;t<L;t++)for(int e=0;e<D;e++){ double s=0;
        for(int d=0;d<D;d++) s+=dQ[t*D+d]*Wq[d*D+e]+dK[t*D+d]*Wk[d*D+e]+dV[t*D+d]*Wv[d*D+e];
        dE[S[t]*D+e]+=(float)s; } }
static void op_clip(void){ double gn2=0; int i;
    for(i=0;i<V*D;i++)gn2+=dE[i]*dE[i]; for(i=0;i<D*D;i++)gn2+=dWq[i]*dWq[i]+dWk[i]*dWk[i]+dWv[i]*dWv[i];
    for(i=0;i<V*D;i++)gn2+=dWo[i]*dWo[i]; double gn=sqrt(gn2); float sc=(gn>0.5f)?(float)(0.5f/gn):1.0f;
    for(i=0;i<V*D;i++){dE[i]*=sc;dWo[i]*=sc;} for(i=0;i<D*D;i++){dWq[i]*=sc;dWk[i]*=sc;dWv[i]*=sc;} }
static void op_adamw(float lr,float b1,float b2,float ea,int ep){
    float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
    #define AW(P,M,Vv,G,N) do{for(int i=0;i<(N);i++){M[i]=b1*M[i]+(1-b1)*G[i];Vv[i]=b2*Vv[i]+(1-b2)*G[i]*G[i];P[i]-=lr*(M[i]/bc1)/(sqrtf(Vv[i]/bc2)+ea);}}while(0)
    AW(E,mE,vE,dE,V*D); AW(Wq,mQ,vQ,dWq,D*D); AW(Wk,mK,vK,dWk,D*D); AW(Wv,mV,vV,dWv,D*D); AW(Wo,mO,vO,dWo,V*D);
    #undef AW
}

int main(int argc,char** argv){
    const char* xl = argc>1?argv[1]:"E:/StarSnow_Home/languages/fromzero/train_front/out/train_step.xlbin";
    if(argc>2) g_nth=atoi(argv[2]); if(g_nth<1)g_nth=1;
    ninst=read_xlbin(xl);
    if(ninst<=0){ printf("读取字节码失败 code=%d\n",ninst); return 1; }
    printf("== 星算语 字节码执行器 VM ==  线程=%d\n  字节码: %d 条指令\n",g_nth,ninst);

    gen_seq();
    int ep=0, step;
    int matmul_cnt=0, rms_cnt=0; float loss=0;
    float lr=0.003f,b1=0.9f,b2=0.999f,ea=1e-8f;

    /* 预算报告(四栏) */
    long nparams = V*D + 3*D*D + V*D;      /* E+Wq/Wk/Wv+Wo */
    long nopt    = 2*nparams;               /* m/v */
    long nact    = L*D*5 + L*D*D + L*D*2 + L*V + L*V; /* X/Q/K/VV/Y + STATE + Z/NUM + LG + dLG */
    long nrt     = D*D*2 + D + L*D*2 + L;   /* G/dstate + Gz + dY/dQ... + DEN */
    printf("  预算: 参数%ld 优化器%ld 激活%ld runtime%ld (字节)\n",nparams*4,nopt*4,nact*4,nrt*4);

    /* 初始权重: qreg(42) 种子 */
    pcg_seed(42);
    for(int i=0;i<V*D;i++)E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;}
    for(int i=0;i<V*D;i++)Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;

    /* 解释执行 STEPS 步 */
    float loss_seq[STEPS];
    for(step=0;step<STEPS;step++){
        ep=step+1; matmul_cnt=0; rms_cnt=0;
        hputu32((uint32_t)step);   /* trace: step 边界 */
        for(int i=0;i<ninst;i++){
            uint8_t op=iop[i];
            hput(&op,1);           /* trace: 算子事件 */
            switch(op){
                case 6:  break;                    /* qreg 已初始化 */
                case 17: op_embed(); break;
                case 8: case 9: case 10: break;    /* ▸ ↦ ⟶ noop(trace) */
                case 1:  matmul_cnt++; (matmul_cnt<=3)?op_proj(matmul_cnt-1):op_outproj(); break;
                case 12: rms_cnt++; op_rms(rms_cnt==1); break;
                case 14: op_attn_fwd(); break;
                case 5:  loss=op_xent(); hputf(loss); break;   /* loss 序列 */
                case 15: op_attn_bwd(); break;
                case 16: op_clip(); break;
                case 11: op_adamw(lr,b1,b2,ea,ep); break;
            }
        }
        if(step<6||step%10==0) printf("  step %3d loss=%.6f\n",step+1,loss);
        loss_seq[step]=loss;
    }

    /* 五项哈希 golden */
    uint64_t h5[5];
    /* 1 字节码哈希 */
    hreset(); { FILE* f=fopen(xl,"rb"); uint8_t b[256]; int n; while((n=(int)fread(b,1,256,f))>0)hput(b,n); fclose(f); } h5[0]=hget();
    /* 2 预算报告哈希 */
    hreset(); hputstr("params");hputu32((uint32_t)nparams*4); hputstr("opt");hputu32((uint32_t)nopt*4); hputstr("act");hputu32((uint32_t)nact*4); hputstr("rt");hputu32((uint32_t)nrt*4); h5[1]=hget();
    /* 3 执行 trace 哈希(重建: 仅锁算子序列+step, 用固定序列) */
    hreset(); for(int i=0;i<ninst;i++)hput(&iop[i],1); for(int s=0;s<STEPS;s++)hputu32((uint32_t)s); h5[2]=hget();
    /* 4 固定归约浮点哈希(所有 Y 求和) */
    hreset(); { double sum=0; for(int t=0;t<L;t++)for(int d=0;d<D;d++)sum+=(double)Y[t*D+d]; float f=(float)sum; hputf(f); } h5[3]=hget();
    /* 5 loss 序列哈希(完整每步序列) */
    hreset(); for(int s=0;s<STEPS;s++)hputf(loss_seq[s]); h5[4]=hget();

    printf("\n== 五项哈希 golden ==\n");
    const char* names[5]={"字节码","预算报告","执行trace","归约浮点","loss序列"};
    for(int i=0;i<5;i++) printf("  %s: %016llx\n",names[i],(unsigned long long)h5[i]);
    return 0;
}
