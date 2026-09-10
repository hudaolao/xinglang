/*
 * 星算语 fromzero · P0-1 C核分块 matmul 基准
 * 目标:验证确定性并行——单线程与多线程逐位一致(块内行优先累加,每行 k 递增顺序固定);
 *      输出资源签名(FLOPs / GFLOPS / 内存 / golden 浮点哈希)。
 * 编译: gcc -O2 -fopenmp -mavx2 -o matmul_bench matmul_bench.c
 * 架构对齐:v0.1 语义规范 §5 确定性规则(块内固定累加树,块间固定块序,禁按线程完成序归约)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>

#define N 512
#define BS 32

static float A[N*N], B[N*N], Cs[N*N], Cp[N*N];

/* 确定性分块 matmul
 * 并行化外层 jj 块;每个 C[i][j] 的 k 累加顺序固定(k 递增),不同 jj 块不共享 C 元素,
 * 因此单线程与多线程输出逐位一致。
 */
static void matmul(float* C, const float* A, const float* B, int n, int bs, int parallel) {
    memset(C, 0, (size_t)n*n*sizeof(float));
    #pragma omp parallel for schedule(static) if(parallel)
    for (int jj = 0; jj < n; jj += bs) {
        int je = jj + bs < n ? jj + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int ke = kk + bs < n ? kk + bs : n;
            for (int ii = 0; ii < n; ii += bs) {
                int ie = ii + bs < n ? ii + bs : n;
                for (int j = jj; j < je; j++) {
                    for (int k = kk; k < ke; k++) {
                        float b = B[k*n + j];
                        for (int i = ii; i < ie; i++)
                            C[i*n + j] += A[i*n + k] * b;
                    }
                }
            }
        }
    }
}

static void init_fill(float* M, unsigned* s) {
    for (int i = 0; i < N*N; i++) {
        *s = *s * 1103515245u + 12345u;
        M[i] = (float)((*s >> 8) & 0xffff) / 65535.0f - 0.5f;
    }
}

int main(void) {
    unsigned s = 12345u;
    init_fill(A, &s);
    init_fill(B, &s);

    double t0 = omp_get_wtime();
    matmul(Cs, A, B, N, BS, 0);          /* 单线程 */
    double t1 = omp_get_wtime();
    matmul(Cp, A, B, N, BS, 1);          /* 多线程 */
    double t2 = omp_get_wtime();

    int bits_equal = (memcmp(Cs, Cp, sizeof(Cs)) == 0);
    double maxerr = 0.0;
    for (int i = 0; i < N*N; i++) {
        double e = fabs((double)Cs[i] - (double)Cp[i]);
        if (e > maxerr) maxerr = e;
    }

    double flops = 2.0 * N * N * N;
    double membytes = 3.0 * N * N * 4.0;

    printf("P0-1 C核分块 matmul 基准\n");
    printf("  N=%d  BS=%d  线程数=%d\n", N, BS, omp_get_max_threads());
    printf("  逐位一致(bits): %s\n", bits_equal ? "YES" : "NO");
    printf("  最大数值差:     %.3e\n", maxerr);
    printf("  单线程:         %.3f ms  %.2f GFLOPS\n", (t1-t0)*1e3, flops/(t1-t0)/1e9);
    printf("  多线程:         %.3f ms  %.2f GFLOPS\n", (t2-t1)*1e3, flops/(t2-t1)/1e9);
    printf("  资源签名: FLOPs=%.3e  内存=%.1f MB  块=%dx%d\n", flops, membytes/1e6, BS, BS);

    uint64_t h = 1469598103934665603ULL;   /* FNV-1a */
    for (int i = 0; i < N*N; i++) {
        uint32_t b; memcpy(&b, &Cp[i], 4);
        h ^= b; h *= 1099511628211ULL;
    }
    printf("  golden 浮点哈希: %016llx\n", (unsigned long long)h);

    return bits_equal ? 0 : 1;
}
