// 星算语 fromzero · P2 训练闭环·第2块:上下文语言模型
// 从 bigram(只看当前)升级:用前面 CTX 个 token 的 onehot 拼接作特征,预测下一个
// 数据: 斐波那契式序列 x[t]=(x[t-1]+x[t-2]) mod V —— 必须看上下文才能猜对
// 确定性: 合同PCG32 + 固定归约序
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define V 64        /* 词表 */
#define CTX 3       /* 上下文窗口 */
#define D (CTX*V)   /* 特征维度 = 192 */
#define LEN 256     /* 序列长 */
#define EPOCHS 80

static uint64_t pcg_state;
static void pcg_seed(uint64_t s){ pcg_state = s + 1442695040888963407ULL; }
static uint32_t pcg_next(void){
    pcg_state = pcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rot = (uint32_t)(pcg_state >> 59);
    uint64_t xs = (pcg_state >> 18) ^ pcg_state;
    return (uint32_t)(((xs >> (32 - rot)) | (xs << rot)) & 0xFFFFFFFF);
}

static void softmax_row(const float* x, float* y, int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx)mx=x[i];
    double sum=0; for(int i=0;i<n;i++){ y[i]=(float)exp((double)x[i]-(double)mx); sum+=y[i]; }
    for(int i=0;i<n;i++) y[i]=(float)((double)y[i]/sum);
}

int main(void){
    int s[LEN]; pcg_seed(42); s[0]=pcg_next()%V; s[1]=pcg_next()%V;
    for(int i=2;i<LEN;i++) s[i]=(s[i-1]+s[i-2])%V;   /* 斐波那契式 */

    /* 特征 onehot 拼接(CTX 个 token) */
    static float x[D];
    /* W[V*D] + adamw 状态 */
    static float W[V*D], m[V*D], v[V*D], dW[V*D], logits[V], prob[V];
    pcg_seed(7);
    for(int i=0;i<V*D;i++) W[i]=(float)(pcg_next()>>8)/16777216.0f*0.1f;

    float lr=0.05f,b1=0.9f,b2=0.999f,eps=1e-8f;
    printf("== P2 上下文语言模型 (词表V=%d, ctx=%d, 特征维度=%d) ==\n", V, CTX, D);
    printf("规律: x[t]=(x[t-1]+x[t-2]) mod %d —— 须看上下文\n", V);

    for(int ep=1; ep<=EPOCHS; ep++){
        for(int i=0;i<V*D;i++) dW[i]=0;
        double loss_sum=0; int cnt=0;
        for(int t=CTX; t<LEN; t++){
            /* 特征: ctx 个 token 的 onehot 拼接 */
            for(int d=0;d<D;d++) x[d]=0;
            for(int c=0;c<CTX;c++) x[c*V + s[t-CTX+c]] = 1.0f;
            int target = s[t];
            /* logits = W @ x */
            for(int j=0;j<V;j++){
                double acc=0; for(int d=0;d<D;d++) acc += (double)W[j*D+d]*x[d];
                logits[j]=(float)acc;
            }
            softmax_row(logits, prob, V);
            double p = prob[target]; if(p<1e-12) p=1e-12;
            loss_sum += -log(p); cnt++;
            /* 梯度累积: dW[j*D+d] += (prob[j]-onehot) * x[d] */
            for(int j=0;j<V;j++) for(int d=0;d<D;d++)
                dW[j*D+d] += (prob[j]-(j==target?1.0f:0.0f)) * x[d];
        }
        float bc1=1.0f-powf(b1,(float)ep), bc2=1.0f-powf(b2,(float)ep);
        for(int i=0;i<V*D;i++){
            m[i]=b1*m[i]+(1-b1)*dW[i];
            v[i]=b2*v[i]+(1-b2)*dW[i]*dW[i];
            W[i]-=lr*(m[i]/bc1)/(sqrtf(v[i]/bc2)+eps);
        }
        if(ep==1||ep%10==0||ep==EPOCHS)
            printf("  epoch %3d  loss=%.6f\n", ep, (float)(loss_sum/cnt));
    }
    return 0;
}
