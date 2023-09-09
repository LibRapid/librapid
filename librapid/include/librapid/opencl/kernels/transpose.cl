#define TRANSPOSEX 3
#define TRANSPOSEY 3

#define TRANSPOSE_KERNEL_IMPL(DTYPE, TILE_DIM)                                                     \
    __kernel void transpose_##DTYPE(__global DTYPE *out,                                           \
                                    __global const DTYPE *in,                                      \
                                    const int rows,                                                \
                                    const int cols,                                                \
                                    DTYPE alpha) {                                                 \
        __local DTYPE tile[TILE_DIM][TILE_DIM + 1];                                                \
                                                                                                   \
        int x = get_group_id(0) * TILE_DIM + get_local_id(0);                                      \
        int y = get_group_id(1) * TILE_DIM + get_local_id(1);                                      \
                                                                                                   \
        if (x < cols && y < rows) { tile[get_local_id(1)][get_local_id(0)] = in[y * cols + x]; }   \
        barrier(CLK_LOCAL_MEM_FENCE);                                                              \
                                                                                                   \
        x = get_group_id(1) * TILE_DIM + get_local_id(0);                                          \
        y = get_group_id(0) * TILE_DIM + get_local_id(1);                                          \
                                                                                                   \
        if (x < rows && y < cols) {                                                                \
            out[y * rows + x] = tile[get_local_id(0)][get_local_id(1)] * alpha;                    \
        }                                                                                          \
    }

TRANSPOSE_KERNEL_IMPL(int8_t, 16)
TRANSPOSE_KERNEL_IMPL(int16_t, 16)
TRANSPOSE_KERNEL_IMPL(int32_t, 16)
TRANSPOSE_KERNEL_IMPL(int64_t, 16)
TRANSPOSE_KERNEL_IMPL(float, 16)
TRANSPOSE_KERNEL_IMPL(double, 16)
