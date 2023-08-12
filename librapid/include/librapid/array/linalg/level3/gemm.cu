#define TS 32 // Tile size

template<typename Int, typename Alpha, typename TypeA, typename TypeB, typename Beta,
         typename TypeC>
__global__ void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, TypeA *a, Int lda,
                     TypeB *b, Int ldb, Beta beta, TypeC *c, Int ldc) {
    const Int inx = blockIdx.x * blockDim.x + threadIdx.x;
    const Int iny = blockIdx.y * blockDim.y + threadIdx.y;
    const Int ibx = threadIdx.x;
    const Int iby = threadIdx.y;

    __shared__ TypeA Asub[TS][TS];
    __shared__ TypeB Bsub[TS][TS];

    TypeC acc = 0;

    const Int numTiles = (k + TS - 1) / TS;

    for (Int t = 0; t < numTiles; t++) {
        const Int tiledIndex = t * TS + ibx;

        Asub[iby][ibx] = (tiledIndex < k && iny < m)
                           ? (transA ? a[tiledIndex + lda * iny] : a[iny * lda + tiledIndex])
                           : 0.0f;
        Bsub[iby][ibx] = (tiledIndex < k && inx < n)
                           ? (transB ? b[tiledIndex + ldb * inx] : b[iny * ldb + tiledIndex])
                           : 0.0f;

        __syncthreads();

        for (Int j = 0; j < TS; j++) {
            if (t * TS + j < k) { acc += Asub[iby][j] * Bsub[j][ibx]; }
        }

        __syncthreads();
    }

    if (iny < m && inx < n) { c[(iny * ldc) + inx] = alpha * acc + beta * c[(iny * ldc) + inx]; }
}
