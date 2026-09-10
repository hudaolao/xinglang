// 星算语 fromzero · P2 训练闭环·第1块:迷你语言模型(下一token预测)
// bigram 模型 W[V][V]: 输入当前token -> onehot行 -> softmax -> 预测下一token
// 数据: 合同PCG32生成确定性序列(模式:下个=当前+1) —— 基座要学会这种语言规律
// 确定性: PCG32(seed=42/7) + 固定归约序
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 32    /* 词表 */
#define L 8     /* 序列长 */
#define EPOCHS 60

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static void softmax_row(const float* x, float* y, int n){
    float mx = x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double sum = 0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}

int main(void){
    int seq[L]; pcg_seed(42); int t = pcg_next()%V;
    for(int i=0;i<L;i++){ seq[i]=t; t=(t+1)%V; }

    static float W[V*V], m[V*V], v[V*V], dW[V*V], prob[V];
    pcg_seed(7);
    for(int i=0;i<V*V;i++) W[i] = (float)(pcg_next()>>8)/16777216.0f*0.1f;

    float lr=0.1f, b1=0.9f, b2=0.999f, eps=1e-8f;
    printf("== P2 训练闭环·迷你语言模型 (词表V=%d, 序列L=%d) ==\n", V, L);
    printf("数据序列: "); for(int i=0;i<L;i++) printf("%d ",seq[i]); printf(" (规律: 下个=当前+1 mod %d)\n",V);

    for(int ep=1; ep<=EPOCHS; ep++){
        for(int i=0;i<V*V;i++) dW[i]=0;
        double loss_sum=0; int cnt=0;
        /* 前向+梯度累积(batch=序列) */
        for(int i=0;i<L-1;i++){
            int in=seq[i], target=seq[i+1];
            softmax_row(&W[in*V], prob, V);
            double p = prob[target]; if(p<1e-12) p=1e-12;
            loss_sum += -log(p); cnt++;
            for(int j=0;j<V;j++) dW[in*V+j] += prob[j] - (j==target?1.0f:0.0f);
        }
        /* adamw 更新 */
        float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
        for(int i=0;i<V*V;i++){
            m[i]=b1*m[i]+(1-b1)*dW[i];
            v[i]=b2*v[i]+(1-b2)*dW[i]*dW[i];
            W[i]-=lr*(m[i]/bc1)/(sqrtf(v[i]/bc2)+eps);
        }
        if(ep==1 || ep%10==0 || ep==EPOCHS)
            printf("  epoch %3d  loss=%.6f\n", ep, (float)(loss_sum/cnt));
    }
    return 0;
}
