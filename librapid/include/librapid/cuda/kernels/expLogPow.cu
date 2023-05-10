template<typename Destination, typename Data>
__global__ void expArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = exp(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void logArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = log(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void log2Arrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = log2(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void log10Arrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = log10(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void sqrtArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = sqrt(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void cbrtArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = cbrt(src[kernelIndex]); }
}

template<typename Destination, typename Data>
__global__ void absArrays(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = abs(src[kernelIndex]); }
}

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
