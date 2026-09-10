// 数值梯度验证: 因果线性注意力 反向是否正确
// 对代表性参数做有限差分,对比解析梯度
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 32
#define D 32
#define L 16

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
static float X[L*D], Q[L*D], K[L*D], VV[L*D], Y[L*D], LG[L*V];
static float STATE[L*D*D], Z[L*D], NUM[L*D], DEN[L];
static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
static float elu(float x){ return x>0 ? x : expm1f(x); }
static float elu_p(float x){ return x>0 ? 1.0f : expf(x); }

static void init(void){
    gen_seq(); pcg_seed(7);
    for(int i=0;i<V*D;i++) E[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
    for(int i=0;i<D*D;i++){ Wq[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wk[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; Wv[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f; }
    for(int i=0;i<V*D;i++) Wo[i]=(float)(pcg_next()>>8)/16777216.0f*0.2f;
}
static void forward(void){
    float eps=1e-3f;
    for(int t=0;t<L;t++) for(int d=0;d<D;d++) X[t*D+d]=E[S[t]*D+d];
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        double sq=0,sk=0,sv=0;
        for(int e=0;e<D;e++){ sq+=(double)Wq[d*D+e]*X[t*D+e]; sk+=(double)Wk[d*D+e]*X[t*D+e]; sv+=(double)Wv[d*D+e]*X[t*D+e]; }
        Q[t*D+d]=(float)sq; K[t*D+d]=(float)sk; VV[t*D+d]=(float)sv;
    }
    float st[D*D]; for(int i=0;i<D*D;i++) st[i]=0;
    float z[D]; for(int i=0;i<D;i++) z[i]=0;
    for(int t=0;t<L;t++){
        for(int i=0;i<D;i++){ float ki=elu(K[t*D+i])+1.0f;
            for(int j=0;j<D;j++) st[i*D+j]+=ki*VV[t*D+j];
            z[i]+=ki; }
        for(int i=0;i<D*D;i++) STATE[t*D*D+i]=st[i];
        for(int i=0;i<D;i++) Z[t*D+i]=z[i];
        for(int j=0;j<D;j++){ double s=0; for(int i=0;i<D;i++) s+=(double)st[i*D+j]*Q[t*D+i]; NUM[t*D+j]=(float)s; }
        double de=eps; for(int i=0;i<D;i++) de+=(double)z[i]*Q[t*D+i];
        DEN[t]=(float)de;
        for(int j=0;j<D;j++) Y[t*D+j]=NUM[t*D+j]/DEN[t];
    }
    for(int t=0;t<L;t++) for(int j=0;j<V;j++){
        double s=0; for(int d=0;d<D;d++) s+=(double)Wo[j*D+d]*Y[t*D+d];
        LG[t*V+j]=(float)s;
    }
}
static double loss_and_dLG(float dLG[L*V]){
    double loss=0;
    for(int t=0;t<L-1;t++){ float p[V]; softmax_row(&LG[t*V],p,V);
        int target=S[t+1]; double q=(double)p[target]; if(q<1e-12)q=1e-12; loss+=-log(q);
        for(int j=0;j<V;j++) dLG[t*V+j]=(j==target)?(p[j]-1.0f):p[j]; }
    return loss;
}
static void backward(const float dLG[L*V], float dE[V*D], float dWo[V*D], float dWq[D*D], float dWk[D*D], float dWv[D*D]){
    static float dY[L*D], dQ[L*D], dK[L*D], dV[L*D];
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
        for(int i=0;i<D;i++){
            double dqi=(double)((float)(dden*Z[t*D+i]));
            for(int j=0;j<D;j++) dqi+=(double)dnum[j]*STATE[t*D*D+i*D+j];
            dQ[t*D+i]=(float)dqi;
        }
        for(int i=0;i<D;i++) Gz[i]+=(float)(dden*Q[t*D+i]);
        for(int i=0;i<D;i++) for(int j=0;j<D;j++)
            dstate[i*D+j]=G[i*D+j]+dnum[j]*Q[t*D+i];
        for(int i=0;i<D;i++){ double dk=Gz[i]; for(int j=0;j<D;j++) dk+=(double)dstate[i*D+j]*VV[t*D+j]; dK[t*D+i]=(float)dk; }
        for(int j=0;j<D;j++){ double dv=0; for(int i=0;i<D;i++) dv+=(double)dstate[i*D+j]*(elu(K[t*D+i])+1.0f); dV[t*D+j]=(float)dv; }
        for(int i=0;i<D*D;i++) G[i]=dstate[i];
    }
    for(int t=0;t<L;t++) for(int d=0;d<D;d++){
        float ek=elu_p(K[t*D+d]);
        for(int e=0;e<D;e++){
            dWk[d*D+e]+=dK[t*D+d]*ek*X[t*D+e];
            dWq[d*D+e]+=dQ[t*D+d]*X[t*D+e];
            dWv[d*D+e]+=dV[t*D+d]*X[t*D+e];
        }
    }
    for(int t=0;t<L;t++) for(int e=0;e<D;e++){ double s=0;
        for(int d=0;d<D;d++){
            double gk = dK[t*D+d]*elu_p(K[t*D+d]);
            s+=(double)dQ[t*D+d]*Wq[d*D+e] + gk*Wk[d*D+e] + (double)dV[t*D+d]*Wv[d*D+e];
        }
        dE[S[t]*D+e]+=(float)s; }
}

#define NUMCHECK(NAME,ARR,IDX) do{ \
    double e=1e-3; float o=ARR[IDX]; \
    ARR[IDX]=(float)(o+e); forward(); static float dl1[L*V]; double lp=loss_and_dLG(dl1); \
    ARR[IDX]=(float)(o-e); forward(); double lm=loss_and_dLG(dl1); \
    ARR[IDX]=o; forward(); \
    double num=(lp-lm)/(2*e); double ana=(double)(ARR2_##NAME[IDX]); \
    printf("  %s[%d]: numeric=%.8f analytic=%.8f\n", #NAME, IDX, num, ana); \
}while(0)

int main(void){
    init(); forward();
    static float dLG[L*V]; double base=loss_and_dLG(dLG);
    static float dE[V*D], dWo[V*D], dWq[D*D], dWk[D*D], dWv[D*D];
    backward(dLG, dE, dWo, dWq, dWk, dWv);
    printf("base loss=%.8f\n", base);
    printf("== 数值梯度 vs 解析梯度 (有限差分 h=1e-3) ==\n");
    double e=1e-3;
    #define DIFF(A) do{ double eps_=e; float o=A; A=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); A=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); A=o; forward(); double num=(lp-lm)/(2*eps_); printf("  %-8s[%3d] num=%.8f  ana=%.8f\n", #A, 0, num, (double)dE[0]); }while(0)
    /* E 抽查 */
    { int idx=0; double eps_=e; float o=E[idx]; E[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); E[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); E[idx]=o; forward(); printf("  E   [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dE[idx]); }
    { int idx=5; double eps_=e; float o=E[idx]; E[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); E[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); E[idx]=o; forward(); printf("  E   [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dE[idx]); }
    { int idx=100; double eps_=e; float o=E[idx]; E[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); E[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); E[idx]=o; forward(); printf("  E   [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dE[idx]); }
    /* Wo 抽查 */
    { int idx=0; double eps_=e; float o=Wo[idx]; Wo[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); Wo[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); Wo[idx]=o; forward(); printf("  Wo  [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dWo[idx]); }
    { int idx=17; double eps_=e; float o=Wo[idx]; Wo[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); Wo[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); Wo[idx]=o; forward(); printf("  Wo  [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dWo[idx]); }
    /* Wq/Wk/Wv 抽查 */
    { int idx=0; double eps_=e; float o=Wq[idx]; Wq[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); Wq[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); Wq[idx]=o; forward(); printf("  Wq  [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dWq[idx]); }
    { int idx=300; double eps_=e; float o=Wk[idx]; Wk[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); Wk[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); Wk[idx]=o; forward(); printf("  Wk  [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dWk[idx]); }
    { int idx=500; double eps_=e; float o=Wv[idx]; Wv[idx]=(float)(o+eps_); forward(); static float d1[L*V]; double lp=loss_and_dLG(d1); Wv[idx]=(float)(o-eps_); forward(); double lm=loss_and_dLG(d1); Wv[idx]=o; forward(); printf("  Wv  [%3d] num=%.8f  ana=%.8f\n", idx, (lp-lm)/(2*eps_), (double)dWv[idx]); }
    return 0;
}
