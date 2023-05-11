
#define ABS_KERNEL(DTYPE)                                                                          \
	__kernel void absArrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {             \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = (data[gid] >= 0) ? data[gid] : -data[gid];                                      \
	}

#define ABS_IMPL                                                                                   \
	ABS_KERNEL(int8_t)                                                                             \
	ABS_KERNEL(uint8_t)                                                                            \
	ABS_KERNEL(int16_t)                                                                            \
	ABS_KERNEL(uint16_t)                                                                           \
	ABS_KERNEL(int32_t)                                                                            \
	ABS_KERNEL(uint32_t)                                                                           \
	ABS_KERNEL(int64_t)                                                                            \
	ABS_KERNEL(uint64_t)                                                                           \
	ABS_KERNEL(float)                                                                              \
	ABS_KERNEL(double)

ABS_IMPL
