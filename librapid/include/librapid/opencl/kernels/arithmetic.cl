#define ARITHMETIC_KERNEL(NAME, OP, DTYPE)                                                         \
	__kernel void NAME##Arrays_##DTYPE(                                                            \
	  __global DTYPE *dst, __global const DTYPE *lhs, __global const DTYPE *rhs) {                 \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = lhs[gid] OP rhs[gid];                                                           \
	}                                                                                              \
                                                                                                   \
	__kernel void NAME##ArraysScalarRhs_##DTYPE(                                                   \
	  __global DTYPE *dst, __global const DTYPE *lhs, DTYPE rhs) {                                 \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = lhs[gid] OP rhs;                                                                \
	}                                                                                              \
                                                                                                   \
	__kernel void NAME##ArraysScalarLhs_##DTYPE(                                                   \
	  __global DTYPE *dst, DTYPE lhs, __global const DTYPE *rhs) {                                 \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = lhs OP rhs[gid];                                                                \
	}

#define ARITHMETIC_OP_IMPL(NAME, OP)                                                               \
	ARITHMETIC_KERNEL(NAME, OP, int8_t)                                                            \
	ARITHMETIC_KERNEL(NAME, OP, int16_t)                                                           \
	ARITHMETIC_KERNEL(NAME, OP, int32_t)                                                           \
	ARITHMETIC_KERNEL(NAME, OP, int64_t)                                                           \
	ARITHMETIC_KERNEL(NAME, OP, uint8_t)                                                           \
	ARITHMETIC_KERNEL(NAME, OP, uint16_t)                                                          \
	ARITHMETIC_KERNEL(NAME, OP, uint32_t)                                                          \
	ARITHMETIC_KERNEL(NAME, OP, uint64_t)                                                          \
	ARITHMETIC_KERNEL(NAME, OP, float)                                                             \
	ARITHMETIC_KERNEL(NAME, OP, double)

ARITHMETIC_OP_IMPL(add, +)
ARITHMETIC_OP_IMPL(sub, -)
ARITHMETIC_OP_IMPL(mul, *)
ARITHMETIC_OP_IMPL(div, /)

ARITHMETIC_OP_IMPL(lessThan, <)
ARITHMETIC_OP_IMPL(lessThanEqual, <=)
ARITHMETIC_OP_IMPL(greaterThan, >)
ARITHMETIC_OP_IMPL(greaterThanEqual, >=)
ARITHMETIC_OP_IMPL(elementWiseEqual, ==)
ARITHMETIC_OP_IMPL(elementWiseNotEqual, !=)
