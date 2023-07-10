#define GEMV_IMPL(TYPE)                                                                            \
	__kernel void gemv_##TYPE(const int trans,                                                    \
							  const int32_t M,                                                     \
							  const int32_t N,                                                     \
							  const TYPE alpha,                                                    \
							  __global const TYPE *A,                                              \
							  const int32_t lda,                                                   \
							  __global const TYPE *x,                                              \
							  const int32_t incX,                                                  \
							  const TYPE beta,                                                     \
							  __global TYPE *y,                                                    \
							  const int32_t incy) {                                                \
		/* Get global thread ID */                                                                 \
		int idx = get_global_id(0);                                                                \
                                                                                                   \
		/* Only valid threads perform computation */                                               \
		if (idx < M) {                                                                             \
			/* Compute dot product for this thread's row of matrix A */                            \
			TYPE acc = 0;                                                                          \
			if (trans == 0) {                                                                     \
				/* Non-transposed matrix */                                                        \
				for (int j = 0; j < N; ++j) { acc += A[idx * lda + j] * x[j * incX]; }             \
			} else {                                                                               \
				/* Transposed matrix */                                                            \
				for (int j = 0; j < N; ++j) { acc += A[j * lda + idx] * x[j * incX]; }             \
			}                                                                                      \
			/* Apply alpha scaling to acc and beta scaling to y[idx] then sum */                   \
			y[idx * incy] = alpha * acc + beta * y[idx * incy];                                    \
		}                                                                                          \
	}

GEMV_IMPL(int8_t)
GEMV_IMPL(int16_t)
GEMV_IMPL(int32_t)
GEMV_IMPL(int64_t)
GEMV_IMPL(uint8_t)
GEMV_IMPL(uint16_t)
GEMV_IMPL(uint32_t)
GEMV_IMPL(uint64_t)
GEMV_IMPL(float)
GEMV_IMPL(double)
