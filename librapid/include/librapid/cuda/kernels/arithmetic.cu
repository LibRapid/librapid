#include "kernelHelper.hpp"
#include <cstdint>

template<typename Destination, typename LHS, typename RHS>
__global__ void addArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] + rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void subArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] - rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void mulArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] * rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void divArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] / rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] < rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] > rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] <= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] >= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] == rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseNotEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] != rhs[kernelIndex]; }
}
