
#define TRIGONOMETRY_KERNEL(NAME, DTYPE)                                                           \
    __kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = NAME(data[gid]);                                                                \
    }

#define TRIGONOMETRY_KERNEL_CAST(NAME, DTYPE)                                                      \
    __kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
        int gid  = get_global_id(0);                                                               \
        dst[gid] = (DTYPE)NAME((double)data[gid]);                                                 \
    }

#define TRIG_IMPL(NAME)                                                                            \
    TRIGONOMETRY_KERNEL_CAST(NAME, int8_t)                                                         \
    TRIGONOMETRY_KERNEL_CAST(NAME, uint8_t)                                                        \
    TRIGONOMETRY_KERNEL_CAST(NAME, int16_t)                                                        \
    TRIGONOMETRY_KERNEL_CAST(NAME, uint16_t)                                                       \
    TRIGONOMETRY_KERNEL_CAST(NAME, int32_t)                                                        \
    TRIGONOMETRY_KERNEL_CAST(NAME, uint32_t)                                                       \
    TRIGONOMETRY_KERNEL_CAST(NAME, int64_t)                                                        \
    TRIGONOMETRY_KERNEL_CAST(NAME, uint64_t)                                                       \
    TRIGONOMETRY_KERNEL(NAME, float)                                                               \
    TRIGONOMETRY_KERNEL(NAME, double)

TRIG_IMPL(sin)
TRIG_IMPL(cos)
TRIG_IMPL(tan)
TRIG_IMPL(asin)
TRIG_IMPL(acos)
TRIG_IMPL(atan)
TRIG_IMPL(sinh)
TRIG_IMPL(cosh)
TRIG_IMPL(tanh)
TRIG_IMPL(asinh)
TRIG_IMPL(acosh)
TRIG_IMPL(atanh)
