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

#define DUAL_ARITHMETIC_OP(DTYPE)                                                                  \
	__kernel void addArrays_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,                        \
										 __global const struct Dual_##DTYPE *lhs,                  \
										 __global const struct Dual_##DTYPE *rhs) {                \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs[gid].value + rhs[gid].value;                                     \
		dst[gid].derivative = lhs[gid].derivative + rhs[gid].derivative;                           \
	}                                                                                              \
                                                                                                   \
	__kernel void addArraysScalarRhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  __global const struct Dual_##DTYPE *lhs,         \
												  struct Dual_##DTYPE rhs) {                       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs[gid].value + rhs.value;                                          \
		dst[gid].derivative = lhs[gid].derivative + rhs.derivative;                                \
	}                                                                                              \
                                                                                                   \
	__kernel void addArraysScalarLhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  struct Dual_##DTYPE lhs,                         \
												  __global const struct Dual_##DTYPE *rhs) {       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs.value + rhs[gid].value;                                          \
		dst[gid].derivative = lhs.derivative + rhs[gid].derivative;                                \
	}                                                                                              \
                                                                                                   \
	__kernel void subArrays_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,                        \
										 __global const struct Dual_##DTYPE *lhs,                  \
										 __global const struct Dual_##DTYPE *rhs) {                \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs[gid].value - rhs[gid].value;                                     \
		dst[gid].derivative = lhs[gid].derivative - rhs[gid].derivative;                           \
	}                                                                                              \
                                                                                                   \
	__kernel void subArraysScalarRhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  __global const struct Dual_##DTYPE *lhs,         \
												  struct Dual_##DTYPE rhs) {                       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs[gid].value - rhs.value;                                          \
		dst[gid].derivative = lhs[gid].derivative - rhs.derivative;                                \
	}                                                                                              \
                                                                                                   \
	__kernel void subArraysScalarLhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  struct Dual_##DTYPE lhs,                         \
												  __global const struct Dual_##DTYPE *rhs) {       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs.value - rhs[gid].value;                                          \
		dst[gid].derivative = lhs.derivative - rhs[gid].derivative;                                \
	}                                                                                              \
                                                                                                   \
	__kernel void mulArrays_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,                        \
										 __global const struct Dual_##DTYPE *lhs,                  \
										 __global const struct Dual_##DTYPE *rhs) {                \
		int gid		   = get_global_id(0);                                                         \
		dst[gid].value = lhs[gid].value * rhs[gid].value;                                          \
		dst[gid].derivative =                                                                      \
		  lhs[gid].derivative * rhs[gid].value + lhs[gid].value * rhs[gid].derivative;             \
	}                                                                                              \
                                                                                                   \
	__kernel void mulArraysScalarRhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  __global const struct Dual_##DTYPE *lhs,         \
												  struct Dual_##DTYPE rhs) {                       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs[gid].value * rhs.value;                                          \
		dst[gid].derivative = lhs[gid].derivative * rhs.value + lhs[gid].value * rhs.derivative;   \
	}                                                                                              \
                                                                                                   \
	__kernel void mulArraysScalarLhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  struct Dual_##DTYPE lhs,                         \
												  __global const struct Dual_##DTYPE *rhs) {       \
		int gid				= get_global_id(0);                                                    \
		dst[gid].value		= lhs.value * rhs[gid].value;                                          \
		dst[gid].derivative = lhs.derivative * rhs[gid].value + lhs.value * rhs[gid].derivative;   \
	}                                                                                              \
                                                                                                   \
	__kernel void divArrays_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,                        \
										 __global const struct Dual_##DTYPE *lhs,                  \
										 __global const struct Dual_##DTYPE *rhs) {                \
		int gid		   = get_global_id(0);                                                         \
		dst[gid].value = lhs[gid].value / rhs[gid].value;                                          \
		dst[gid].derivative =                                                                      \
		  (lhs[gid].derivative * rhs[gid].value - lhs[gid].value * rhs[gid].derivative) /          \
		  (rhs[gid].value * rhs[gid].value);                                                       \
	}                                                                                              \
                                                                                                   \
	__kernel void divArraysScalarRhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  __global const struct Dual_##DTYPE *lhs,         \
												  struct Dual_##DTYPE rhs) {                       \
		int gid		   = get_global_id(0);                                                         \
		dst[gid].value = lhs[gid].value / rhs.value;                                               \
		dst[gid].derivative =                                                                      \
		  (lhs[gid].derivative * rhs.value - lhs[gid].value * rhs.derivative) /                    \
		  (rhs.value * rhs.value);                                                                 \
	}                                                                                              \
                                                                                                   \
	__kernel void divArraysScalarLhs_Dual_##DTYPE(__global struct Dual_##DTYPE *dst,               \
												  struct Dual_##DTYPE lhs,                         \
												  __global const struct Dual_##DTYPE *rhs) {       \
		int gid		   = get_global_id(0);                                                         \
		dst[gid].value = lhs.value / rhs[gid].value;                                               \
		dst[gid].derivative =                                                                      \
		  (lhs.derivative * rhs[gid].value - lhs.value * rhs[gid].derivative) /                    \
		  (rhs[gid].value * rhs[gid].value);                                                       \
	}

DUAL_ARITHMETIC_OP(int8_t)
DUAL_ARITHMETIC_OP(int16_t)
DUAL_ARITHMETIC_OP(int32_t)
DUAL_ARITHMETIC_OP(int64_t)
DUAL_ARITHMETIC_OP(uint8_t)
DUAL_ARITHMETIC_OP(uint16_t)
DUAL_ARITHMETIC_OP(uint32_t)
DUAL_ARITHMETIC_OP(uint64_t)
DUAL_ARITHMETIC_OP(float)
DUAL_ARITHMETIC_OP(double)
