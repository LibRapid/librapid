template<typename Destination, typename Data>
__global__ void absArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = abs(src[kernelIndex]); }
}
