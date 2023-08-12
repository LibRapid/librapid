template<typename Destination, typename Data>
__global__ void sigmoidActivationForward(size_t elements, Destination *dst, Data *src) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = 1 / (1 + exp((float)-src[kernelIndex])); }
}

template<>
__global__ void sigmoidActivationForward(size_t elements, float *dst, float *src) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = 1 / (1 + exp(-src[kernelIndex])); }
}

template<>
__global__ void sigmoidActivationForward(size_t elements, double *dst, double *src) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = 1 / (1 + exp(-src[kernelIndex])); }
}

template<typename Destination, typename Data>
__global__ void sigmoidActivationBackward(size_t elements, Destination *dst, Data *src) {
    const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (kernelIndex < elements) { dst[kernelIndex] = src[kernelIndex] * (1 - src[kernelIndex]); }
}
