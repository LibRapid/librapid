template<typename Destination, typename Source>
__global__ void fillArray(size_t elements, Destination *dst, Source value) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = value; }
}