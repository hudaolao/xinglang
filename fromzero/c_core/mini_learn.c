// 星算语 fromzero · P1 迷你学习闭环(内核自我生长)
// logits -> softmax -> cross_entropy -> 梯度回传 -> adamw 更新
// 跑若干步,loss 应下降:证明内核不仅能算单个算子,还能"学"
// 确定性:合同PCG32(seed=42)初始化 + 固定归约序
#include <stdio.h>
#include <math.h>
#include <stdint.h>

/* 合同 PCG32(与 P0-5 一致,64位 xorshifted) */
static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static void softmax(const float* x, float* y, size_t n){
    float mx = x[0]; for(size_t i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum = 0; for(size_t i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(size_t i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}

int main(void){
    enum { N = 8, STEPS = 8 };
    float logits[N], prob[N], m[N]={0}, v[N]={0};
    pcg_seed(42);
    for(int i=0;i<N;i++) logits[i] = (float)(pcg_next()>>8)/16777216.0f;
    size_t target = 3;
    float lr = 0.5f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.0f;

    printf("== 星算语迷你学习闭环 (target=%zu, 确定性seed=42) ==\n", target);
    float first = -1;
    for(int step=1; step<=STEPS; step++){
        softmax(logits, prob, N);
        double p = (double)prob[target]; if(p<1e-12) p=1e-12;
        float loss = -(float)log(p);
        if(step==1) first = loss;
        /* 梯度: dL/dlogits = prob - onehot */
        float grad[N];
        for(int i=0;i<N;i++) grad[i] = prob[i] - (i==(int)target ? 1.0f : 0.0f);
        /* adamw 更新 */
        float bc1 = 1.0f-powf(b1,(float)step), bc2 = 1.0f-powf(b2,(float)step);
        for(int i=0;i<N;i++){
            float gi = grad[i] - wd*logits[i];
            m[i] = b1*m[i] + (1-b1)*gi;
            v[i] = b2*v[i] + (1-b2)*gi*gi;
            logits[i] -= lr * (m[i]/bc1) / (sqrtf(v[i]/bc2)+eps);
        }
        printf("  step %d  loss=%.6f  P[target]=%.4f\n", step, loss, prob[target]);
    }
    printf("loss 收敛: %.4f -> (最后一步 %.4f) %s\n", first,
           (float)(-log((double)prob[target] + 1e-12)),
           (prob[target]>0.98f) ? "学会 ✓" : "学习中");
    return 0;
}
