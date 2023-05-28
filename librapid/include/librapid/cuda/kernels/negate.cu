// Note: errors in this file will appear on the wrong line, since we copy another header file
//       in to provide some utility functions (the include paths in Jitify are somewhat unreliable)

template<typename Destination, typename DATA>
__global__ void negateArrays(size_t elements, Destination *dst, DATA *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = -data[kernelIndex]; }
}

template<>
__global__ void negateArrays(size_t elements, float2 *dst, float2 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
	}
}

template<>
__global__ void negateArrays(size_t elements, float3 *dst, float3 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
		dst[kernelIndex].z = -data[kernelIndex].z;
	}
}

template<>
__global__ void negateArrays(size_t elements, float4 *dst, float4 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
		dst[kernelIndex].z = -data[kernelIndex].z;
		dst[kernelIndex].w = -data[kernelIndex].w;
	}
}

template<>
__global__ void negateArrays(size_t elements, double2 *dst, double2 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
	}
}

template<>
__global__ void negateArrays(size_t elements, double3 *dst, double3 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
		dst[kernelIndex].z = -data[kernelIndex].z;
	}
}

template<>
__global__ void negateArrays(size_t elements, double4 *dst, double4 *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) {
		dst[kernelIndex].x = -data[kernelIndex].x;
		dst[kernelIndex].y = -data[kernelIndex].y;
		dst[kernelIndex].z = -data[kernelIndex].z;
		dst[kernelIndex].w = -data[kernelIndex].w;
	}
}
