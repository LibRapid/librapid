
#define SIGMOID_KERNEL(DTYPE)                                                                      \
	__kernel void sigmoidActivation_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {     \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = 1 / (1 + exp(-data[gid]));                                                      \
	}                                                                                              \
                                                                                                   \
	__kernel void sigmoidDerivative_##DTYPE(__global DTYPE *dst, __global const DTYPE *data) {     \
		int gid	 = get_global_id(0);                                                               \
		dst[gid] = data[gid] * (1 - data[gid]);                                                    \
	}                                                                                              \
                                                                                                   \
	struct _sigmoid_semicolon_forcer_##DTYPE {}

#define KERNEL_IMPL(NAME)                                                                          \
	NAME##_KERNEL(float);                                                                          \
	NAME##_KERNEL(double);

KERNEL_IMPL(SIGMOID)
