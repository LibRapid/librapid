#define TS 32 // Tile size

template<typename Int, typename Alpha, typename TypeA, typename Beta, typename TypeX,
         typename TypeY>
__global__ void gemv(bool trans, Int m, Int n, Alpha alpha, TypeA *a, Int lda, TypeX *x, Int incx,
                     Beta beta, TypeY *y, Int incy) {
    const Int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        TypeY acc = 0;

        if (trans) {
            for (Int i = 0; i < n; i++) { acc += a[idx + i * lda] * x[i * incx]; }
        } else {
            for (Int i = 0; i < n; i++) { acc += a[i + idx * lda] * x[i * incx]; }
        }

        y[idx * incy] = alpha * acc + beta * y[idx * incy];
    }
}
