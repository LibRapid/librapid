#define FLOORCEILROUND_KERNEL(NAME, DTYPE)                                                         \
	__kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = NAME(data[gid]);                                                                \
	}

#define FLOORCEILROUND_KERNEL_CAST(NAME, DTYPE)                                                    \
	__kernel void NAME##Arrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = data[gid];                                                                      \
	}

#define FLOORCEILROUND_IMPL(NAME)                                                                  \
	FLOORCEILROUND_KERNEL_CAST(NAME, int8_t)                                                       \
	FLOORCEILROUND_KERNEL_CAST(NAME, uint8_t)                                                      \
	FLOORCEILROUND_KERNEL_CAST(NAME, int16_t)                                                      \
	FLOORCEILROUND_KERNEL_CAST(NAME, uint16_t)                                                     \
	FLOORCEILROUND_KERNEL_CAST(NAME, int32_t)                                                      \
	FLOORCEILROUND_KERNEL_CAST(NAME, uint32_t)                                                     \
	FLOORCEILROUND_KERNEL_CAST(NAME, int64_t)                                                      \
	FLOORCEILROUND_KERNEL_CAST(NAME, uint64_t)                                                     \
	FLOORCEILROUND_KERNEL(NAME, float)                                                             \
	FLOORCEILROUND_KERNEL(NAME, double)

FLOORCEILROUND_IMPL(floor)
FLOORCEILROUND_IMPL(ceil)
