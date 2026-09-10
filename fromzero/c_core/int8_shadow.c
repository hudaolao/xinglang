// 星算语 fromzero · P1 INT8 量化乘加影子基准
// 量化路径:不进 golden 主哈希,只报告量化误差上界(诚实记录,不污染确定性基线)
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define M 16
#define N 16
#define K 16

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

/* fp32 参考 matmul(确定性) */
static void matmul_fp32(const float* A, const float* B, float* C){
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){
        double s=0; for(int k=0;k<K;k++) s += (double)A[i*K+k]*(double)B[k*N+j];
        C[i*N+j]=(float)s;
    }
}
/* int8 量化 matmul:量化到[-127,127],int32 累加,反量化 */
static void matmul_int8(const float* A, const float* B, float* C, float sA, float sB){
    int8_t aq[M*K], bq[K*N];
    for(int i=0;i<M*K;i++){ int v=(int)roundf(A[i]/sA); if(v>127)v=127; if(v<-127)v=-127; aq[i]=(int8_t)v; }
    for(int i=0;i<K*N;i++){ int v=(int)roundf(B[i]/sB); if(v>127)v=127; if(v<-127)v=-127; bq[i]=(int8_t)v; }
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){
        int32_t acc=0; for(int k=0;k<K;k++) acc += (int32_t)aq[i*K+k]*(int32_t)bq[k*N+j];
        C[i*N+j]=(float)((double)acc*sA*sB);
    }
}

int main(void){
    static float A[M*K], B[K*N], Cf[M*N], Ci[M*N];
    pcg_seed(42);
    for(int i=0;i<M*K;i++) A[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;
    for(int i=0;i<K*N;i++) B[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;

    /* 量化 scale:max|.|/127 */
    float maxA=0,maxB=0;
    for(int i=0;i<M*K;i++){ float a=fabsf(A[i]); if(a>maxA)maxA=a; }
    for(int i=0;i<K*N;i++){ float b=fabsf(B[i]); if(b>maxB)maxB=b; }
    float sA=maxA/127.0f, sB=maxB/127.0f;

    matmul_fp32(A,B,Cf);
    matmul_int8(A,B,Ci,sA,sB);

    double max_abs=0, max_rel=0, fp_max=0;
    for(int i=0;i<M*N;i++){
        double d=fabs((double)Cf[i]-(double)Ci[i]);
        double r=d/(fabs((double)Cf[i])+1e-9);
        if(d>max_abs)max_abs=d;
        if(r>max_rel)max_rel=r;
        if(fabs((double)Cf[i])>fp_max)fp_max=fabs((double)Cf[i]);
    }
    printf("== P1 INT8 量化影子基准 (M=N=K=%d, seed=42) ==\n",M);
    printf("fp32 参考峰值 |C|: %.6f\n", fp_max);
    printf("最大绝对误差   : %.6f (相对峰值 %.2f%%)\n", max_abs, max_abs/fp_max*100.0);
    printf("最大相对误差   : %.6f\n", max_rel);
    printf("量化路径不进入 golden 主哈希(诚实记录,不污染确定性基线)\n");
    return 0;
}
