#define SIGMOID_KERNEL(DTYPE)                                                                      \
    __kernel void sigmoidActivationForward_##DTYPE(__global DTYPE *dst,                            \
                                                   __global const DTYPE *data) {                   \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = 1 / (1 + exp(-data[gid]));                                                      \
    }                                                                                              \
                                                                                                   \
    __kernel void sigmoidActivationBackward_##DTYPE(__global DTYPE *dst,                           \
                                                    __global const DTYPE *data) {                  \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = data[gid] * (1 - data[gid]);                                                    \
    }                                                                                              \
                                                                                                   \
    struct _sigmoid_semicolon_forcer_##DTYPE {}

#define SIGMOID_KERNEL_CAST(DTYPE)                                                                 \
    __kernel void sigmoidActivationForward_##DTYPE(__global DTYPE *dst,                            \
                                                   __global const DTYPE *data) {                   \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = 1 / (1 + exp((float)-data[gid]));                                               \
    }                                                                                              \
                                                                                                   \
    __kernel void sigmoidActivationBackward_##DTYPE(__global DTYPE *dst,                           \
                                                    __global const DTYPE *data) {                  \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = data[gid] * (1 - data[gid]);                                                    \
    }                                                                                              \
                                                                                                   \
    struct _sigmoid_semicolon_forcer_##DTYPE {}

#define KERNEL_IMPL(NAME)                                                                          \
    NAME##_KERNEL_CAST(int8_t);                                                                    \
    NAME##_KERNEL_CAST(int16_t);                                                                   \
    NAME##_KERNEL_CAST(int32_t);                                                                   \
    NAME##_KERNEL_CAST(int64_t);                                                                   \
    NAME##_KERNEL_CAST(uint8_t);                                                                   \
    NAME##_KERNEL_CAST(uint16_t);                                                                  \
    NAME##_KERNEL_CAST(uint32_t);                                                                  \
    NAME##_KERNEL_CAST(uint64_t);                                                                  \
    NAME##_KERNEL(float);                                                                          \
    NAME##_KERNEL(double);

KERNEL_IMPL(SIGMOID)
