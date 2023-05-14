// Note: errors in this file will appear on the wrong line, since we copy another header file
//       in to provide some utility functions (the include paths in Jitify are somewhat unreliable)

template<typename Destination, typename DATA>
__global__ void negateArrays(size_t elements, Destination *dst, DATA *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = -data[kernelIndex]; }
}
