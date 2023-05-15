#define NEGATE_KERNEL(DTYPE)                                                                       \
	__kernel void negateArrays_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {          \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = -data[gid];                                                                     \
	}

NEGATE_KERNEL(int8_t)
NEGATE_KERNEL(uint8_t)
NEGATE_KERNEL(int16_t)
NEGATE_KERNEL(uint16_t)
NEGATE_KERNEL(int32_t)
NEGATE_KERNEL(uint32_t)
NEGATE_KERNEL(int64_t)
NEGATE_KERNEL(uint64_t)
NEGATE_KERNEL(float)
NEGATE_KERNEL(double)
