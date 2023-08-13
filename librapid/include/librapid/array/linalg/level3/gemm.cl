#define TS 32 // Tile size

#define GEMM_IMPL(TYPE)                                                                            \
    __kernel void gemm_##TYPE(const int transA,                                                    \
                              const int transB,                                                    \
                              const int32_t M,                                                     \
                              const int32_t N,                                                     \
                              const int32_t K,                                                     \
                              const TYPE alpha,                                                    \
                              __global TYPE *A,                                                    \
                              const int32_t lda,                                                   \
                              __global const TYPE *B,                                              \
                              const int32_t ldb,                                                   \
                              const TYPE beta,                                                     \
                              __global TYPE *C,                                                    \
                              const int32_t ldc) {                                                 \
        const int32_t inx = get_global_id(0);                                                      \
        const int32_t iny = get_global_id(1);                                                      \
        const int32_t ibx = get_local_id(0);                                                       \
        const int32_t iby = get_local_id(1);                                                       \
                                                                                                   \
        __local TYPE Asub[TS][TS];                                                                 \
        __local TYPE Bsub[TS][TS];                                                                 \
                                                                                                   \
        TYPE acc = 0;                                                                              \
                                                                                                   \
        const int32_t numTiles = K / TS + 1;                                                       \
                                                                                                   \
        for (int32_t t = 0; t < numTiles; t++) {                                                   \
            const int32_t tiledIndex = t * TS + ibx;                                               \
                                                                                                   \
            Asub[iby][ibx] = (tiledIndex < K && iny < M)                                           \
                               ? (transA ? A[tiledIndex + lda * iny] : A[iny * lda + tiledIndex])  \
                               : 0.0f;                                                             \
            Bsub[iby][ibx] = (tiledIndex < K && inx < N)                                           \
                               ? (transB ? B[tiledIndex + ldb * inx] : B[iny * ldb + tiledIndex])  \
                               : 0.0f;                                                             \
                                                                                                   \
            barrier(CLK_LOCAL_MEM_FENCE);                                                          \
                                                                                                   \
            for (int32_t k = 0; k < TS; k++) {                                                     \
                if (t * TS + k < K) { acc += Asub[iby][k] * Bsub[k][ibx]; }                        \
            }                                                                                      \
                                                                                                   \
            barrier(CLK_LOCAL_MEM_FENCE);                                                          \
        }                                                                                          \
                                                                                                   \
        if (iny < M && inx < N) {                                                                  \
            C[(iny * ldc) + inx] = alpha * acc + beta * C[(iny * ldc) + inx];                      \
        }                                                                                          \
    }

GEMM_IMPL(int8_t)
GEMM_IMPL(int16_t)
GEMM_IMPL(int32_t)
GEMM_IMPL(int64_t)
GEMM_IMPL(uint8_t)
GEMM_IMPL(uint16_t)
GEMM_IMPL(uint32_t)
GEMM_IMPL(uint64_t)
GEMM_IMPL(float)
GEMM_IMPL(double)
