#include <stdint.h>

__global__ template<typename Source, typename Destination>
void fillArray(size_t elements, Source *src, Destination *dst) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernekIndex < elements) { dst[kernekIndex] = src[kernekIndex]; }
}