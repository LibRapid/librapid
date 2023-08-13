#define EXPLOGPOW_KERNEL(NAME, DTYPE)                                                              \
    __kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = NAME(data[gid]);                                                                \
    }

#define ABS_KERNEL(DTYPE)                                                                          \
    __kernel void absArrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {             \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = (data[gid] >= 0) ? data[gid] : -data[gid];                                      \
    }

#define EXPLOGPOW_KERNEL_CAST(NAME, DTYPE)                                                         \
    __kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = (DTYPE)NAME((double)data[gid]);                                                 \
    }

#define EXPLOGPOW_IMPL(NAME)                                                                       \
    EXPLOGPOW_KERNEL_CAST(NAME, int8_t)                                                            \
    EXPLOGPOW_KERNEL_CAST(NAME, uint8_t)                                                           \
    EXPLOGPOW_KERNEL_CAST(NAME, int16_t)                                                           \
    EXPLOGPOW_KERNEL_CAST(NAME, uint16_t)                                                          \
    EXPLOGPOW_KERNEL_CAST(NAME, int32_t)                                                           \
    EXPLOGPOW_KERNEL_CAST(NAME, uint32_t)                                                          \
    EXPLOGPOW_KERNEL_CAST(NAME, int64_t)                                                           \
    EXPLOGPOW_KERNEL_CAST(NAME, uint64_t)                                                          \
    EXPLOGPOW_KERNEL(NAME, float)                                                                  \
    EXPLOGPOW_KERNEL(NAME, double)

EXPLOGPOW_IMPL(exp)
EXPLOGPOW_IMPL(log)
EXPLOGPOW_IMPL(log2)
EXPLOGPOW_IMPL(log10)
EXPLOGPOW_IMPL(sqrt)
EXPLOGPOW_IMPL(cbrt)
