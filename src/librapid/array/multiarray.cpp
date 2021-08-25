#include <librapid/array/multiarray.hpp>

namespace librapid
{
	void Array::fill(double val)
	{
		AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
					   m_extent.size(), val);
	}

	void Array::fill(const Complex<double> &val)
	{
		AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
					   m_extent.size(), val);
	}

	Array Array::operator+(const Array &other) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		Array res(m_extent, m_dtype, m_location);

		AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						m_extent.size(), ops::Add(), "add");

		return res;
	}

	void Array::add(const Array &other, Array &res) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						m_extent.size(), ops::Add(), "add");
	}

	Array Array::operator-(const Array &other) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		Array res(m_extent, m_dtype, m_location);

		AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						m_extent.size(), ops::Sub(), "sub");

		return res;
	}

	void Array::sub(const Array &other, Array &res) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						m_extent.size(), ops::Sub(), "sub");
	}

	template<typename A, typename B, typename C, class FUNC>
	void Array::simpleCPUop(librapid::Accelerator locnA,
							librapid::Accelerator locnB,
							librapid::Accelerator locnC,
							const A *a, const B *b, C *c, size_t size,
							const FUNC &op, const std::string &name)
	{
		if (locnA == Accelerator::CPU && locnB == Accelerator::CPU && locnC == Accelerator::CPU)
		{
		#pragma omp parallel for shared(a, b, c, size, op) num_threads(4) default(none)
			for (lr_int i = 0; i < (lr_int) size; ++i)
				c[i] = (C) op(a[i], b[i]);
		}
	#ifdef LIBRAPID_HAS_CUDA
		else
		{
			using jitify::reflection::Type;
			using jitify::reflection::type_of;

			// const char *simpleKernel = R"V0G0N(adder
			// 	__constant__ long long LIBRAPID_MAX_DIMS = 32;
			// 	template<typename A, typename B, typename C, typename LAMBDA>
			// 	__global__
			// 	void op(const A *a, const B *b, C *c,
			// 			 size_t size, LAMBDA func) {
			// 		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
			// 		// if (tid < size) c[tid] = a[tid] + b[tid];
			// 		if (tid < size) c[tid] = func(a[tid], b[tid]);
			// 	}
			// 	)V0G0N";

			std::string kernel = R"V0G0N(adder
			__constant__ long long LIBRAPID_MAX_DIMS = 32;
			template<typename A, typename B, typename C>
			__global__
			void function(const A *arrayA, const B *arrayB, C *arrayC, size_t size) {
				unsigned int kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;

				if (kernelIndex < size) {
					const auto &a = arrayA[kernelIndex];
					const auto &b = arrayB[kernelIndex];
					auto &c = arrayC[kernelIndex];
			)V0G0N";

			kernel += op.kernel;

			kernel += "\n}\n}";

			static jitify::JitCache kernel_cache;
			jitify::Program program = kernel_cache.program(kernel, 0);

			unsigned int threadsPerBlock, blocksPerGrid;

			// Use 1 to 512 threads per block
			if (size < 256)
			{
				threadsPerBlock = (unsigned int) size;
				blocksPerGrid = 1;
			}
			else
			{
				threadsPerBlock = 256;
				blocksPerGrid = ceil(double(size) / double(threadsPerBlock));
			}

			dim3 grid(blocksPerGrid);
			dim3 block(threadsPerBlock);

			auto f = [=](const A &a, const B &b)
			{
				return op(a, b);
			};

		#ifdef LIBRAPID_CUDA_STREAM
			jitifyCall(program.kernel("function")
					   .instantiate(Type<A>(), Type<B>(), Type<C>())
					   .configure(grid, block, 0, cudaStream)
					   .launch(a, b, c, size));
		#else
			jitifyCall(program.kernel("op")
					   .instantiate(Type<A>(), Type<B>(), Type<C>())
					   .configure(grid, block)
					   .launch(a, b, c, size));
		#endif // LIBRAPID_CUDA_STREAM

			// auto instance = program.kernel("add").instantiate(Type<A>(),
			// 												  Type<B>(),
			// 												  Type<C>());

			// long long maxDims = LIBRAPID_MAX_DIMS;
			// cuMemcpyHtoD(instance.get_constant_ptr("LIBRAPID_MAX_DIMS"),
			// 			 &maxDims,
			// 			 sizeof(long long));

			// cudaSafeCall(cudaDeviceSynchronize());
			// instance.configure(grid, block);
			// jitifyCall(instance.configure(grid, block).launch(a, b, c, size));
		}
	#endif // LIBRAPID_HAS_CUDA
	}

	template<typename A, typename B, typename C>
	void Array::simpleFill(librapid::Accelerator locnA,
						   librapid::Accelerator locnB,
						   A *data, B *, size_t size,
						   C val)
	{
		if (locnA == Accelerator::CPU)
		{
			for (size_t i = 0; i < size; ++i)
				data[i] = (A) val;
		}
	#ifdef LIBRAPID_HAS_CUDA
		else
		{
			auto tmp = (A *) malloc(sizeof(A) * size);
			for (size_t i = 0; i < size; ++i)
				tmp[i] = (A) val;

			// cudaSafeCall(cudaMemcpyAsync(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice, cudaStream));

		#ifdef LIBRAPID_CUDA_STREAM
			cudaSafeCall(cudaMemcpyAsync(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice, cudaStream));
			cudaSafeCall(cudaStreamSynchronize(cudaStream));
		#else
			cudaSafeCall(cudaDeviceSynchronize());
			cudaSafeCall(cudaMemcpy(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice));
			cudaSafeCall(cudaDeviceSynchronize());
		#endif
			free(tmp);
		}
	#endif
	}
}