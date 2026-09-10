// 星算语 fromzero · P1 确定性并行验证
// 六智者裁决:确定性并行不是单线程——块内固定行优先累加,块间固定二叉归约
// 关键:归约树焊死(固定8块 + 固定二叉序),与线程数无关;线程只改并行度
// 验证:1 线程 vs 8 线程 layernorm 逐位一致
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define N (1<<16)
#define BLOCKS 8   /* 固定分块数,归约树与线程数无关 */
static float X[N], Y1[N], Y8[N];
static double PART[BLOCKS];

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

/* 分层求和:固定8块,块内行优先,块间二叉归约(与线程数无关) */
static double reduce8(const float* x, size_t n, int nt){
    size_t bs = (n + BLOCKS - 1) / BLOCKS;
    /* 线程只循环分配块;每块局部和 = 行优先累加(同一顺序) */
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        for(int b=tid; b<BLOCKS; b+=nt){
            size_t s0=(size_t)b*bs, s1=s0+bs; if(s1>n) s1=n;
            double s=0; for(size_t i=s0;i<s1;i++) s+=x[i];
            PART[b]=s;
        }
    }
    /* 固定二叉归约树 */
    int m=BLOCKS;
    while(m>1){ int h=(m+1)/2; for(int i=0;i<m/2;i++) PART[i]=PART[i]+PART[i+h]; m=h; }
    return PART[0];
}

static double reduce8_2(const float* x, double mean, size_t n, int nt);
static void layernorm(const float* x, float* y, size_t n, float eps, int nt){
    double mean = reduce8(x,n,nt)/(double)n;
    double var = reduce8_2(x,mean,n,nt)/(double)n;
    double inv = 1.0/sqrt(var+eps);
    size_t bs=(n+BLOCKS-1)/BLOCKS;
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        for(int b=tid;b<BLOCKS;b+=nt){
            size_t s0=(size_t)b*bs, s1=s0+bs; if(s1>n)s1=n;
            for(size_t i=s0;i<s1;i++) y[i]=(float)((x[i]-mean)*inv);
        }
    }
}

static double reduce8_2(const float* x, double mean, size_t n, int nt){
    size_t bs=(n+BLOCKS-1)/BLOCKS;
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        for(int b=tid;b<BLOCKS;b+=nt){
            size_t s0=(size_t)b*bs, s1=s0+bs; if(s1>n)s1=n;
            double s=0; for(size_t i=s0;i<s1;i++){ double d=x[i]-mean; s+=d*d; }
            PART[b]=s;
        }
    }
    int m=BLOCKS; while(m>1){ int h=(m+1)/2; for(int i=0;i<m/2;i++) PART[i]=PART[i]+PART[i+h]; m=h; }
    return PART[0];
}

static int bits_equal(const float* a, const float* b, size_t n){
    for(size_t i=0;i<n;i++) if(memcmp(&a[i],&b[i],4)!=0) return 0;
    return 1;
}

int main(void){
    pcg_seed(42);
    for(size_t i=0;i<N;i++) X[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;

    layernorm(X,Y1,N,1e-5f,1);
    layernorm(X,Y8,N,1e-5f,8);
    printf("== P1 确定性并行: layernorm (N=%d, 固定8块二叉归约) ==\n", N);
    printf("1线程 vs 8线程 逐位一致: %s\n", bits_equal(Y1,Y8,N)?"YES":"NO");
    if(bits_equal(Y1,Y8,N)){
        /* golden 摘要 */
        uint32_t h=2166136261u;
        for(size_t i=0;i<N;i++){ uint32_t b; memcpy(&b,&Y8[i],4); h^=b; h*=16777619u; }
        printf("layernorm golden(8线程) = %08x\n", h);
    }
    return 0;
}
