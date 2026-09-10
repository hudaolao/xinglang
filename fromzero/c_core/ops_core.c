// 星算语 fromzero · P1 C核执行内核：核心算子（确定性）
// 算子: add / layernorm / softmax / cross_entropy / adamw
// 确定性:固定归约序 + 合同PCG32(seed=42)输入 + FNV-1a golden哈希
// 编译: gcc -O2 -o ops_core ops_core.c
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* ---- 合同 PCG32 (P0-5 qreg_seed_contract v0.1) ---- */
static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);   /* 0..31 */
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;  /* 64位,不截断 */
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

/* ---- FNV-1a 浮点哈希（与 matmul_bench 一致）---- */
static uint32_t fnv1a(const float* v, size_t n, uint32_t h){
    for(size_t i=0;i<n;i++){ uint32_t b; memcpy(&b,&v[i],4); h^=b; h*=16777619u; }
    return h;
}

/* ---- 算子（确定性，固定归约序）---- */
/* add: y = x + b */
static void op_add(const float* x, const float* b, float* y, size_t n){
    for(size_t i=0;i<n;i++) y[i]=x[i]+b[i];
}
/* layernorm: 先累加求 mean -> 累加求 var -> 归一化（固定序） */
static void op_layernorm(const float* x, float* y, size_t n, float eps){
    double sum=0; for(size_t i=0;i<n;i++) sum+=x[i];
    double mean=sum/(double)n;
    double var=0; for(size_t i=0;i<n;i++){ double d=x[i]-mean; var+=d*d; }
    var/=(double)n;
    double inv=1.0/sqrt(var+eps);
    for(size_t i=0;i<n;i++) y[i]=(float)((x[i]-mean)*inv);
}
/* softmax: 减max求稳 -> exp -> 固定累加归一 */
static void op_softmax(const float* x, float* y, size_t n){
    float mx=x[0]; for(size_t i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum=0; for(size_t i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(size_t i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}
/* cross_entropy: loss = -log(y[target])（钳制下界） */
static float op_cross_entropy(const float* y, size_t t){
    double p=(double)y[t]; if(p<1e-12)p=1e-12; return -(float)log(p);
}
/* adamw: 权重衰减 + 一/二阶矩 + 偏差修正（确定性，t 从1计） */
static void op_adamw(float* w, const float* g, float* m, float* v, size_t n,
                     float lr, float b1, float b2, float eps, float wd, uint32_t t){
    float bc1=1.0f-powf(b1,(float)t), bc2=1.0f-powf(b2,(float)t);
    for(size_t i=0;i<n;i++){
        float gi=g[i]-wd*w[i];
        m[i]=b1*m[i]+(1-b1)*gi;
        v[i]=b2*v[i]+(1-b2)*gi*gi;
        float mh=m[i]/bc1, vh=v[i]/bc2;
        w[i]-=lr*mh/(sqrtf(vh)+eps);
    }
}

int main(void){
    enum{N=8};
    float x[N],b[N],w[N],g[N],m[N]={0},v[N]={0};

    /* 验证合同 PCG32(42) golden（应=0x7f7aa885 0xaf88544a ...） */
    pcg_seed(42);
    printf("PCG32(42) 前6值: ");
    for(int k=0;k<6;k++) printf("%08x ", pcg_next());
    printf("\n");

    /* 确定性输入（合同 PCG32，seed=42） */
    pcg_seed(42);
    for(int i=0;i<N;i++){
        x[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;
        b[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;
        w[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;
        g[i]=(float)(pcg_next()>>8)/16777216.0f-0.5f;
    }

    float y[N];
    op_add(x,b,y,N);              uint32_t ha=fnv1a(y,N,2166136261u);
    op_layernorm(x,y,N,1e-5f);    uint32_t hl=fnv1a(y,N,2166136261u);
    op_softmax(x,y,N);            uint32_t hs=fnv1a(y,N,2166136261u);
    float loss=op_cross_entropy(y,3);
    op_adamw(w,g,m,v,N,1e-3f,0.9f,0.999f,1e-8f,0.01f,1);
    uint32_t hw=fnv1a(w,N,2166136261u);

    printf("== P1 算子 golden（seed=42, N=8）==\n");
    printf("add         hash = %08x\n", ha);
    printf("layernorm   hash = %08x\n", hl);
    printf("softmax     hash = %08x\n", hs);
    printf("cross_entropy loss = %.8f\n", loss);
    printf("adamw(w)    hash = %08x\n", hw);
    return 0;
}
