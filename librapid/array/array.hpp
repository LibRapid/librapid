#ifndef LIBRAPID_ARRAY
#define LIBRAPID_ARRAY

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/ops.hpp>

#include <atomic>
#include <functional>

#ifdef LIBRAPID_REFCHECK
#define increment() _increment(__LINE__)
#define decrement() _decrement(__LINE__)
#endif // LIBRAPID_REFCHECK

namespace librapid
{
	class Array
	{
	public:
		/**
		 * \rst
		 *
		 * Default constructor for the Array type. It does not initialize any values
		 * and many functions will throw an error when given an empty array.
		 *
		 * \endrst
		 */
		Array()
		{
			initializeCudaStream();
		};

		/**
		 * \rst
		 *
		 * Create a new Array from an Extent and an optional Datatype and
		 * Accelerator. The Extent defines the number of dimensions of the Array,
		 * as well as the size of each dimension.
		 *
		 * Parameters
		 * ----------
		 *
		 * extent: ``Extent``
		 *		The dimensions for the Array
		 * dtype: ``Datatype = FLOAT32``
		 *		The datatype for the Array
		 * location: ``Accelerator = CPU``
		 *
		 * \endrst
		 */
		Array(const Extent &extent, Datatype dtype = Datatype::FLOAT32,
			  Accelerator location = Accelerator::CPU)
		{
			initializeCudaStream();

			if (extent.containsAutomatic())
				throw std::invalid_argument("Cannot create an Array from an Extent"
											" containing automatic values. "
											"Extent was " + extent.str());

			constructNew(extent, Stride::fromExtent(extent), dtype, location);
		}

		/**
		 * \rst
		 *
		 * Create an Array object from an existing one. This constructor copies all
		 * values, and the new Array shares the data of the Array passed to it, so
		 * an update in one will result in an update in the other.
		 *
		 * Note that if the input Array is not initialized, the function will
		 * quick return and not initialize the new Array.
		 *
		 * .. Hint::
		 *
		 *		If you want to create an exact copy of an Array, but don't want the
		 *		data to be linked, see the ``Array::copy()`` function.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: ``Array``
		 *		The Array instance to construct from
		 *
		 * \endrst
		 */
		Array(const Array &other)
		{
			// Quick return if possible
			if (other.m_references == nullptr)
				return;

			m_location = other.m_location;
			m_dtype = other.m_dtype;
			m_dataStart = other.m_dataStart;
			m_dataOrigin = other.m_dataOrigin;

			m_references = other.m_references;

			m_extent = other.m_extent;
			m_stride = other.m_stride;

			m_isScalar = other.m_isScalar;
			m_isChild = false;

			increment();
		}

		LR_INLINE Array &operator=(const Array &other)
		{
			// Quick return if possible
			if (other.m_references == nullptr)
				return *this;

			if (m_references == nullptr)
				constructNew(other.m_extent, other.m_stride,
							 other.m_dtype, other.m_location);

			if (m_isChild && m_extent != other.m_extent)
				throw std::invalid_argument("Cannot set child array with "
											+ m_extent.str() + " to "
											+ other.m_extent.str());

			// Attempt to copy the data from other into *this
			if (m_stride.isContiguous() && other.m_stride.isContiguous())
				AUTOCAST_MEMCPY(makeVoidPtr(), other.makeVoidPtr(),
								m_extent.size());
			else
				throw std::runtime_error("Haven't gotten to this yet...");

			increment();

			return *this;
		}

		~Array()
		{
			decrement();
		}

		template<typename _Ty>
		LR_INLINE void fill(const _Ty &val)
		{
			AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
						   m_extent.size(), val);
		}

		LR_INLINE Array operator+(const Array &other) const
		{
			// Add two arrays together
			if (!(m_stride.isTrivial() && m_stride.isContiguous()))
				throw std::runtime_error("Yeah, you can't do this yet");

			if (m_location != other.m_location)
				throw std::runtime_error("Can't do this either, unfortunately");

			Array res(m_extent, m_dtype, m_location);

			static auto operation = ops::Add();

			AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
							other.makeVoidPtr(), res.makeVoidPtr(),
							m_extent.size(), operation);

			return res;
		}

		LR_INLINE void add(const Array &other, Array &res) const
		{
			// Add two arrays together
			if (!(m_stride.isTrivial() && m_stride.isContiguous()))
				throw std::runtime_error("Yeah, you can't do this yet");

			if (m_location != other.m_location)
				throw std::runtime_error("Can't do this either, unfortunately");

			AUTOCAST_BINARY(simpleCPUop, makeVoidPtr(),
							other.makeVoidPtr(), res.makeVoidPtr(),
							m_extent.size(), ops::Add());
		}

		LR_INLINE std::string str() const
		{
			std::string res;
			try
			{
				if (m_location == Accelerator::CPU)
				{
					AUTOCAST_UNARY(Array::printLinear, makeVoidPtr(), validVoidPtr,
								   m_extent.size(), res);
				}
			#ifdef LIBRAPID_HAS_CUDA
				else
				{
					auto tmp = AUTOCAST_ALLOC(m_dtype, Accelerator::CPU, m_extent.size());
					AUTOCAST_MEMCPY(tmp, makeVoidPtr(), m_extent.size());

					AUTOCAST_UNARY(Array::printLinear, tmp, validVoidPtr,
								   math::min(m_extent.size(), 50), res);

					AUTOCAST_FREE(tmp);
				}
			#endif
			}
			catch (std::exception &e)
			{
				std::cout << "Error: " << e.what() << "\n";
			}

			return res;
		}

	private:
	#ifdef LIBRAPID_REFCHECK
		LR_INLINE void _increment(int line) const
		{
			if (m_references == nullptr)
				return;

			(*m_references)++;

			std::cout << "Incrementing at line " << line << ". References is now "
				<< *m_references << "\n";
		}
	#else
		LR_INLINE void initializeCudaStream() const
		{
		#ifdef LIBRAPID_HAS_CUDA
		#ifdef LIBRAPID_CUDA_STREAM
			if (!streamCreated)
			{
				checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream,
								cudaStreamNonBlocking));
				streamCreated = true;
			}
		#endif // LIBRAPID_CUDA_STREAM
		#endif // LIBRAPID_HAS_CUDA
		}

		LR_INLINE void increment() const
		{
			if (m_references == nullptr)
				return;

			(*m_references)++;
		}
	#endif // LIBRAPID_REFCHECK

	#ifdef LIBRAPID_REFCHECK
		LR_INLINE void _decrement(int line)
		{
			if (m_references == nullptr)
				return;

			(*m_references)--;

			if (*m_references == 0)
			{
				std::cout << "Freeing data at line " << line << "\n";

				// Delete data
				AUTOCAST_FREE({m_dataOrigin, m_dtype, m_location});
				delete m_references;
			}
			else
			{
				printf("Decrementing at line %i. References is now %i\n", line,
					   (int) *m_references);

				std::cout << "Decrementing at line " << line
					<< ". References is now " << *m_references << "\n";
			}
		}
	#else
		LR_INLINE void decrement()
		{
			if (m_references == nullptr)
				return;

			(*m_references)--;

			if (*m_references == 0)
			{
				// Delete data
				AUTOCAST_FREE({m_dataOrigin, m_dtype, m_location});
				delete m_references;
			}
		}
	#endif // LIBRAPID_REFCHECK

		LR_INLINE VoidPtr makeVoidPtr() const
		{
			return {m_dataStart, m_dtype, m_location};
		}

		LR_INLINE void constructNew(const Extent &e, const Stride &s,
									const Datatype &dtype,
									const Accelerator &location)
		{
			// Is scalar if extent is [0]
			bool isScalar = (e.ndim() == 1) && (e[0] == 0);

			// Construct members
			m_location = location;
			m_dtype = dtype;

			// If array is scalar, allocate "sizeof(dtype)" bytes -- e.size() is 0
			m_dataStart = AUTOCAST_ALLOC(dtype, location, e.size() + isScalar).ptr;
			m_dataOrigin = m_dataStart;

			m_references = new std::atomic<size_t>(1);

			m_extent = e;
			m_stride = s;

			m_isScalar = isScalar;
			m_isChild = false;
		}

		template<typename A, typename B, typename C, class FUNC>
		LR_INLINE static void simpleCPUop(librapid::Accelerator locnA,
										  librapid::Accelerator locnB,
										  librapid::Accelerator locnC,
										  A *a, B *b, C *c, size_t size,
										  const FUNC &op)
		{
			if (locnA == Accelerator::CPU && locnB == Accelerator::CPU && locnC == Accelerator::CPU)
			{
			#pragma omp parallel for shared(a, b, c, size, op) num_threads(4) default(none)
				for (lr_int i = 0; i < size; ++i)
					c[i] = op(a[i], b[i]);
			}
		#ifdef LIBRAPID_HAS_CUDA
			else
			{
				using jitify::reflection::Type;

				static const char *program_source =
					"adder\n"
					"__constant__ long long LIBRAPID_MAX_DIMS = 32;\n"
					"template<typename A, typename B, typename C>\n"
					"__global__\n"
					"void add(const A *a, const B *b, C *c, size_t size) {\n"
					"	// unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
					"	// unsigned int stride = blockDim.x * gridDim.x;\n"
					"	// for (unsigned int i = index; i < size; i += stride) {\n"
					"	// 	c[i] = a[i] + b[i];\n"
					"	// }\n"
					"	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n"
					"	if (tid < size) c[tid] = a[tid] + b[tid];\n"
					"}\n";

				static jitify::JitCache kernel_cache;
				jitify::Program program = kernel_cache.program(program_source, 0);

				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (size < 256)
				{
					threadsPerBlock = size;
					blocksPerGrid = 1;
				}
				else
				{
					threadsPerBlock = 256;
					blocksPerGrid = ceil(double(size) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

			#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaStreamSynchronize(cudaStream));
				jitifyCall(program.kernel("add")
						   .instantiate(Type<A>(), Type<B>(), Type<C>())
						   .configure(grid, block, 0, cudaStream)
						   .launch(a, b, c, size));
			#else
				cudaSafeCall(cudaDeviceSynchronize());
				jitifyCall(program.kernel("add")
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
		LR_INLINE static void simpleFill(librapid::Accelerator locnA,
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

				cudaSafeCall(cudaDeviceSynchronize());
				// cudaSafeCall(cudaMemcpyAsync(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice, cudaStream));

			#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaMemcpyAsync(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice, cudaStream));
			#else
				cudaSafeCall(cudaMemcpy(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice));
			#endif

				free(tmp);
			}
		#endif
		}

		template<typename A, typename B>
		LR_INLINE static void printLinear(librapid::Accelerator locnA,
										  librapid::Accelerator locnB,
										  A *data, B *, size_t size,
										  std::string &res)
		{
			std::stringstream tmp;
			for (size_t i = 0; i < size; ++i)
				tmp << data[i] << ", ";
			res = tmp.str();
		}

	private:
		Accelerator m_location = Accelerator::CPU;
		Datatype m_dtype = Datatype::NONE;

		void *m_dataStart = nullptr;
		void *m_dataOrigin = nullptr;

		// std::atomic to allow for multithreading, because multiple threads may
		// increment/decrement at the same clock cycle, resulting in values being
		// incorrect and errors turning up.
		std::atomic<size_t> *m_references = nullptr;

		Extent m_extent;
		Stride m_stride;

		bool m_isScalar = false; // Array is a scalar value
		bool m_isChild = false; // Array is a direct subscript of another (e.g. x[0])
	};
}

#ifdef LIBRAPID_REFCHECK
#undef increment
#undef decrement
#endif // LIBRAPID_REFCHECK

#endif // LIBRAPID_ARRAY