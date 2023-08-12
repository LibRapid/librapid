// Note: errors in this file will appear on the wrong line, since we copy another header file
//       in to provide some utility functions (the include paths in Jitify are somewhat unreliable)

template<typename Destination, typename LHS, typename RHS>
__global__ void addArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] + rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void addArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) {
        dst[kernelIndex] = lhs[kernelIndex] + rhs;
        // printf("%d + %d = %d\n", lhs[kernelIndex], rhs, dst[kernelIndex]);
    }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void addArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs + rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void subArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] - rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void subArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs - rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void subArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] - rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void mulArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] * rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void mulArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs * rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void mulArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] * rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void divArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] / rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void divArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs / rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void divArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] / rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] < rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs < rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] < rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] > rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs > rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] > rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] <= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanEqualArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs, RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] <= rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void lessThanEqualArraysScalarLhs(size_t elements, Destination *dst, LHS lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs <= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] >= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanEqualArraysScalarLhs(size_t elements, Destination *dst, LHS lhs,
                                                RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs >= rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void greaterThanEqualArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs,
                                                RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] >= rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] == rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseEqualArraysScalarLhs(size_t elements, Destination *dst, LHS lhs,
                                                RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs == rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseEqualArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs,
                                                RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] == rhs; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseNotEqualArrays(size_t elements, Destination *dst, LHS *lhs, RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] != rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseNotEqualArraysScalarLhs(size_t elements, Destination *dst, LHS lhs,
                                                   RHS *rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs != rhs[kernelIndex]; }
}

template<typename Destination, typename LHS, typename RHS>
__global__ void elementWiseNotEqualArraysScalarRhs(size_t elements, Destination *dst, LHS *lhs,
                                                   RHS rhs) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = lhs[kernelIndex] != rhs; }
}
