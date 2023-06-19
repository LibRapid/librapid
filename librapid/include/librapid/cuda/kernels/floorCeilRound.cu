
template<typename Destination, typename Data>
__global__ void floorArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = floor(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void ceilArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = ceil(src[kernelIndex]); }
}
