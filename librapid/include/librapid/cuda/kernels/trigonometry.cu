#define TRIG_IMPL(NAME)                                                                            \
	template<typename Destination, typename Data>                                                  \
	__global__ void NAME##Arrays(size_t elements, Destination *dst, Data *data) {                  \
		const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;                          \
		if (kernelIndex < elements) { dst[kernelIndex] = NAME(data[kernelIndex]); }                \
	}

TRIG_IMPL(sin)
TRIG_IMPL(cos)
TRIG_IMPL(tan)
TRIG_IMPL(asin)
TRIG_IMPL(acos)
TRIG_IMPL(atan)
TRIG_IMPL(sinh)
TRIG_IMPL(cosh)
TRIG_IMPL(tanh)
