#include "kernelHelper.hpp"
#include <stdint.h>
#include <stdio.h>

template<typename Destination, typename LHS, typename RHS>
__global__ void addArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] + rhs[kernelIndex]; }
}
