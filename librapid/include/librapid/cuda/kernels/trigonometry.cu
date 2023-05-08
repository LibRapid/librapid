template<typename Destination, typename Data>
__global__ void sinArrays(size_t elements, Destination *dst, Data *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = sin(data[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void cosArrays(size_t elements, Destination *dst, Data *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = cos(data[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void tanArrays(size_t elements, Destination *dst, Data *data) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = tan(data[kernelIndex]); }
}
