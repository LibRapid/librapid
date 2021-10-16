#ifndef LIBRAPID_AUTOCAST
#define LIBRAPID_AUTOCAST

#include <algorithm>
#include <variant>

namespace librapid
{
	/**
	 * \rst
	 *
	 * Contains the different datatypes supported by
	 * the LibRapid AutoCast system
	 *
	 * \endrst
	 */
	enum class Datatype
	{
		NONE,			// no datatype
		VALIDNONE,		// No datatype, but it is not required
		BOOL,			// bool
		INT64,			// int64_t
		FLOAT32,		// float
		FLOAT64,		// double
	};

	/**
	 * \rst
	 *
	 * An instance of the ``std::variant`` type, containing pointers for each of the
	 * supported datatypes.
	 *
	 * \endrst
	 */
	using RawArrayData = std::variant<
		bool *,
		int64_t *,
		float *,
		double *
	>;

	/**
	 * \rst
	 *
	 * Contains valid accelerators, which enable you to store data on the host or
	 * on the device if CUDA support is enabled.
	 *
	 * \endrst
	 */
	enum class Accelerator
	{
		NONE,
		CPU,
		GPU
	};

	/**
	 * \rst
	 *
	 * Contains all the information required for an array. It is wrapped by the
	 * Array class to provide extra functionality.
	 *
	 * \endrst
	 */
	struct RawArray
	{
		RawArrayData data;
		Datatype dtype = Datatype::NONE;
		Accelerator location = Accelerator::CPU;
	};

	/**
	 * \rst
	 *
	 * Returns true if the provided datatype represents an integer value.
	 *
	 * \endrst
	 */
	inline bool isIntegral(Datatype t)
	{
		switch (t)
		{
			case Datatype::NONE:
				return false;
			case Datatype::VALIDNONE:
				return false;
			case Datatype::BOOL:
				return true;
			case Datatype::INT64:
				return true;
			default:
				return false;
		}
	}

	/**
	* \rst
	*
	* Returns true if the provided datatype represents an unsigned value.
	*
	* \endrst
	*/
	inline bool isUnsigned(Datatype t)
	{
		switch (t)
		{
			case Datatype::NONE:
				return false;
			case Datatype::VALIDNONE:
				return false;
			case Datatype::BOOL:
				return false;
			case Datatype::INT64:
				return false;
			default:
				return false;
		}
	}

	/**
	* \rst
	*
	* Returns true if the provided datatype represents a floating point (or complex)
	* value.
	*
	* \endrst
	*/
	inline bool isFloating(Datatype t)
	{
		switch (t)
		{
			case Datatype::FLOAT32:
				return true;
			case Datatype::FLOAT64:
				return true;
			default:
				return false;
		}
	}

	/**
	* \rst
	*
	* Returns the number of bytes of memory needed to store a single element of the
	* provided datatype
	*
	* \endrst
	*/
	inline size_t datatypeBytes(Datatype t)
	{
		switch (t)
		{
			case Datatype::NONE:
				return 0;
			case Datatype::VALIDNONE:
				return 1;
			case Datatype::BOOL:
				return sizeof(bool);
			case Datatype::INT64:
				return sizeof(int64_t);
			case Datatype::FLOAT32:
				return sizeof(float);
			case Datatype::FLOAT64:
				return sizeof(double);
		}

		return 0;
	}

	inline std::string datatypeToString(const Datatype &t)
	{
		switch (t)
		{
			case Datatype::NONE:
				return "NONE";
			case Datatype::VALIDNONE:
				return "VALIDNONE";
			case Datatype::BOOL:
				return "BOOL";
			case Datatype::INT64:
				return "INT64";
			case Datatype::FLOAT32:
				return "FLOAT32";
			case Datatype::FLOAT64:
				return "FLOAT64";
		}

		return "UNKNOWN";
	}

	/**
	 * \rst
	 *
	 * Converts a C++ typename into a LibRapid datatype enum
	 *
	 * \endrst
	 */
	template<typename T>
	inline Datatype typeToDatatype(T x)
	{
		if constexpr (std::is_same<T, bool>::value) return Datatype::BOOL;
		if constexpr (std::is_same<T, int64_t>::value) return Datatype::INT64;
		if constexpr (std::is_same<T, float>::value) return Datatype::FLOAT32;
		if constexpr (std::is_same<T, double>::value) return Datatype::FLOAT64;

		return Datatype::NONE;
	}

	/**
	 * /rst
	 *
	 * Generate a LibRapid datatype from a string.
	 *
	 * The string can be formatted as any of the following:
	 *  - The C++ typename
	 *  - <type><bytes>
	 *		- "int8"
	 *		- "float64"
	 *		- "cfloat32"
	 *		- etc.
	 *  - Shorthand <type><size>
	 *		- "i" -> integer
	 *		- "ui" -> unsigned integer
	 *		- "f" -> floating
	 *		- "cf" -> complex floating point
	 *  - Single specific character
	 *		- "n" -> None
	 *		- "b" -> Bool
	 *		- "i" -> 64-bit signed integer
	 *		- "ui" -> Unsigned 64-bit integer
	 *		- "f" -> 64-bit floating point
	 *		- "c" -> 64-bit complex floating point
	 *
	 * \endrst
	 */
	inline Datatype stringToDatatype(const std::string &str)
	{
		// Force the string to be lower case
		std::string temp = str;
		std::transform(temp.begin(), temp.end(), temp.begin(),
					   [](unsigned char c)
		{
			return std::tolower(c);
		});

		// Different types and their potential string representations
		static std::vector<std::string> noneStr = {
			"n",
			"none",
			"null",
			"void"
		};

		static std::vector<std::string> boolStr = {
			"b",
			"bool",
			"boolean"
		};

		static std::vector<std::string> int8Str = {
			"int8",
			"i8",
			"short"
		};

		static std::vector<std::string> uint8Str = {
			"uint8",
			"ui8",
			"unsigned short"
		};

		static std::vector<std::string> int16Str = {
			"int16",
			"i16",
			"int"
		};

		static std::vector<std::string> uint16Str = {
			"uint16",
			"ui16",
			"unsigned int"
		};

		static std::vector<std::string> int32Str = {
			"int32",
			"i32",
			"long"
		};

		static std::vector<std::string> uint32Str = {
			"uint32",
			"ui32",
			"unsigned long"
		};

		static std::vector<std::string> int64Str = {
			"i",
			"int64",
			"i64",
			"long long"
		};

		static std::vector<std::string> uint64Str = {
			"ui",
			"uint64",
			"ui64",
			"unsigned long long"
		};

		static std::vector<std::string> float32Str = {
			"float32",
			"f32",
			"float"
		};

		static std::vector<std::string> float64Str = {
			"f",
			"float64",
			"f64",
			"double"
		};

		static std::vector<std::string> cfloat32Str = {
			"cfloat32",
			"cf32",
			"complex float"
		};

		static std::vector<std::string> cfloat64Str = {
			"c",
			"cfloat64",
			"cf64",
			"complex double"
		};

		static std::map<Datatype, std::vector<std::string>> types = {
			{Datatype::NONE, noneStr},
			{Datatype::BOOL, boolStr},
			{Datatype::INT64, int64Str},
			{Datatype::FLOAT32, float32Str},
			{Datatype::FLOAT64, float64Str},
		};

		// Locate the datatype
		for (const auto &dtypePair : types)
		{
			for (const auto &name : dtypePair.second)
			{
				if (name == temp)
					return dtypePair.first;
			}
		}

		throw std::invalid_argument("Name \"" + str + "\" is invalid. See "
									"documentation for details and valid inputs");
	}

	/**
	 * \rst
	 *
	 * Convert a string to a LibRapid accelerator enum. Valid inputs are:
	 *  - "none", "null" -> NONE
	 *  - "cpu" -> CPU
	 *	- "gpu" -> GPU
	 *
	 * .. Hint::
	 *		There is no case-checking for the input, so you could use "GpU" if you
	 *		really wanted to
	 *
	 * \endrst
	 */
	inline Accelerator stringToAccelerator(const std::string &str)
	{
		// Force the string to be lower case
		std::string temp = str;
		std::transform(temp.begin(), temp.end(), temp.begin(),
					   [](unsigned char c)
		{
			return std::tolower(c);
		});

		if (temp == "none" ||
			temp == "null")
			return Accelerator::NONE;

		if (temp == "cpu")
			return Accelerator::CPU;

		if (temp == "gpu")
		#ifdef LIBRAPID_HAS_CUDA
			return Accelerator::GPU;
	#else
			throw std::invalid_argument("CUDA support is not enabled, so \"GPU\" is"
										" not a valid accelerator.");
	#endif // LIBRAPID_HAS_CUDA

		throw std::invalid_argument("Accelerator \"" + str + "\" is invalid. See "
									"documentation for details and valid inputs");
	}

#ifdef LIBRAPID_HAS_CUDA
#ifdef LIBRAPID_CUDA_STREAM
	extern cudaStream_t cudaStream;
	extern bool streamCreated;
#endif // LIBRAPID_CUDA_STREAM
#endif // LIBRAPID_HAS_CUDA

	/**
	 * \rst
	 *
	 * Allocate memory for a number of elements of a specific datatype.
	 *
	 * Parameters
	 * ----------
	 *
	 * raw: RawArray
	 *
	 *
	 * \endrst
	 */
	inline RawArray rawArrayMalloc(RawArray &raw, uint64_t elems)
	{
		if (raw.location == Accelerator::CPU)
		{
			switch (raw.dtype)
			{
				case Datatype::BOOL:
					{
						raw.data = (bool *)
							alignedMalloc(sizeof(bool) * elems);
						break;
					}
				case Datatype::INT64:
					{
						raw.data = (int64_t *)
							alignedMalloc(sizeof(int8_t) * elems);
						break;
					}
				case Datatype::FLOAT32:
					{
						raw.data = (float *)
							alignedMalloc(sizeof(float) * elems);
						break;
					}
				case Datatype::FLOAT64:
					{
						raw.data = (double *)
							alignedMalloc(sizeof(double) * elems);
						break;
					}
			}
		}
		else if (raw.location == Accelerator::GPU)
		{
			void *memory = nullptr;
			int64_t bytes = datatypeBytes(raw.dtype) * elems;

		#ifdef LIBRAPID_HAS_CUDA
		#ifdef LIBRAPID_CUDA_STREAM
			cudaSafeCall(cudaMallocAsync(&memory, bytes, cudaStream));
		#else
			cudaSafeCall(cudaMalloc(&memory, datatypeBytes(raw.dtype) * elems));
		#endif
		#endif // LIBRAPID_HAS_CUDA

			switch (raw.dtype)
			{
				case Datatype::BOOL:
					{
						raw.data = (bool *) memory;
						break;
					}
				case Datatype::INT64:
					{
						raw.data = (int64_t *) memory;
						break;
					}
				case Datatype::FLOAT32:
					{
						raw.data = (float *) memory;
						break;
					}
				case Datatype::FLOAT64:
					{
						raw.data = (double *) memory;
						break;
					}
			}
		}
		else
		{
			raw.data = (bool *) nullptr;
		}

		return raw;
	}

	inline void freeRawArray(RawArray raw)
	{
		void *memory = nullptr;

		switch (raw.dtype)
		{
			case Datatype::BOOL:
				{
					memory = std::get<bool *>(raw.data);
					break;
				}
			case Datatype::INT64:
				{
					memory = std::get<int64_t *>(raw.data);
					break;
				}
			case Datatype::FLOAT32:
				{
					memory = std::get<float *>(raw.data);
					break;
				}
			case Datatype::FLOAT64:
				{
					memory = std::get<double *>(raw.data);
					break;
				}
		}

		if (raw.location == Accelerator::CPU)
		{
			alignedFree(memory);
		}
	#ifdef LIBRAPID_HAS_CUDA
		else if (raw.location == Accelerator::GPU)
		{
		#ifdef LIBRAPID_CUDA_STREAM
			cudaSafeCall(cudaFreeAsync(memory, cudaStream));
		#else
			cudaSafeCall(cudaFree(memory));
		#endif
		}
	#else
		throw std::runtime_error("CUDA support was not enabled, so device memory cannot be freed");
	#endif
	}

	inline void rawArrayMemcpy(RawArray &dst,
							   const RawArray &src, uint64_t elems)
	{
		if (dst.location == Accelerator::NONE ||
			src.location == Accelerator::NONE)
			throw std::invalid_argument("Cannot copy to unknown device");

		if ((int) dst.dtype < 2 || (int) src.dtype < 2)
			throw std::invalid_argument("Cannot copy data to or from a null "
										"datatype");

		if (dst.location == src.location &&
			dst.dtype == src.dtype)
		{
			if (dst.dtype == src.dtype)
			{
				// A simple memcpy will suffice, as the datatypes are identical

				std::visit([&](auto *a, auto *b)
				{
					if (src.location == Accelerator::CPU)
					{
						// CPU to CPU memcpy
						memcpy(a, b, datatypeBytes(src.dtype) * elems);
					}
					else
					{
					#ifdef LIBRAPID_HAS_CUDA
					#ifdef LIBRAPID_CUDA_STREAM
						if (src.location == Accelerator::CPU &&
							dst.location == Accelerator::GPU)
							cudaSafeCall(cudaMemcpyAsync(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyHostToDevice, cudaStream));
						else if (src.location == Accelerator::GPU &&
								 dst.location == Accelerator::CPU)
							cudaSafeCall(cudaMemcpyAsync(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyDeviceToHost, cudaStream));
						else if (src.location == Accelerator::GPU &&
								 dst.location == Accelerator::GPU)
							cudaSafeCall(cudaMemcpyAsync(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyDeviceToDevice, cudaStream));
					#else
						if (src.location == Accelerator::CPU &&
							dst.location == Accelerator::GPU)
							cudaSafeCall(cudaMemcpy(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyHostToDevice));
						else if (src.location == Accelerator::GPU &&
								 dst.location == Accelerator::CPU)
							cudaSafeCall(cudaMemcpy(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyDeviceToHost));
						else if (src.location == Accelerator::GPU &
								 dst.location == Accelerator::GPU)
							cudaSafeCall(cudaMemcpy(a, b,
										 datatypeBytes(src.dtype) * elems,
										 cudaMemcpyDeviceToDevice));
					#endif // LIBRAPID_CUDA_STREAMS
					#endif // LIBRAPID_HAS_CUDA
					}
				}, dst.data, src.data);
			}
		}
		else if (dst.location == Accelerator::CPU &&
				 src.location == Accelerator::CPU)
		{
			std::visit([&](auto *a, auto *b)
			{
				using A = typename std::remove_pointer<decltype(a)>::type;
				using B = typename std::remove_pointer<decltype(b)>::type;

				if (elems < THREAD_THREASHOLD)
				{
					for (int64_t i = 0; i < elems; ++i)
						a[i] = b[i];
				}
				else
				{
				#pragma omp parallel for shared(a, b) num_threads(NUM_THREADS)
					for (int64_t i = 0; i < elems; ++i)
						a[i] = b[i];
				}
			}, dst.data, src.data);
		}
	#ifdef LIBRAPID_HAS_CUDA
		else
		{
			if (dst.location != src.location)
			{
				if (src.location == Accelerator::CPU)
				{
					// Copy from CPU to GPU

					std::visit([&](auto *a, auto *b)
					{
						using A = typename std::remove_pointer<decltype(a)>::type;
						using B = typename std::remove_pointer<decltype(b)>::type;

						for (int64_t i = 0; i < elems; ++i)
						{
							A tmpVal = A(b[i]);

						#ifdef LIBRAPID_CUDA_STREAM
							cudaSafeCall(cudaMemcpyAsync(a + i, &tmpVal,
										 sizeof(A), cudaMemcpyHostToDevice,
										 cudaStream));
						#else
							cudaSafeCall(cudaMemcpy(a + i, &tmpVal,
										 sizeof(A), cudaMemcpyHostToDevice));
						#endif
						}
					}, dst.data, src.data);
				}
				else if (src.location == Accelerator::GPU)
				{
					// Copy from GPU to CPU

					std::visit([&](auto *a, auto *b)
					{
						using A = typename std::remove_pointer<decltype(a)>::type;
						using B = typename std::remove_pointer<decltype(b)>::type;

						for (int64_t i = 0; i < elems; ++i)
						{
							B tmp;

						#ifdef LIBRAPID_CUDA_STREAM
							cudaSafeCall(cudaMemcpyAsync(&tmp, b + i,
										 sizeof(B), cudaMemcpyDeviceToHost,
										 cudaStream));
						#else
							cudaSafeCall(cudaMemcpy(&tmp, b + i,
										 sizeof(A), cudaMemcpyDeviceToHost));
						#endif

							a[i] = (A) tmp;
						}
					}, dst.data, src.data);
				}
			}
			else
			{
				// Copy from GPU to GPU

				using jitify::reflection::Type;

				std::string kernel = "copyKernel\n";
				kernel = R"V0G0N(
							template<typename A, typename B>
							__global__
							void copyKernel(const A *__restrict arrayA,
											const B *__restrict arrayB,
											size_t elems)
							{
								uint16_t kernelIndex = blockDim.x * blockIdx.x
														   + threadIdx.x;

								if (kernelIndex < elems) {
									arrayA[kernelIndex] = (A) arrayB[kernelIndex];
								}
							}
							)V0G0N";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, 0);

				uint16_t threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512)
				{
					threadsPerBlock = (uint16_t) elems;
					blocksPerGrid = 1;
				}
				else
				{
					threadsPerBlock = 512;
					blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				// int64_t elems = elems;

				std::visit([&](auto *a, auto *b)
				{
					using A = typename std::remove_pointer<decltype(a)>::type;
					using B = typename std::remove_pointer<decltype(b)>::type;

				#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("copyKernel")
							   .instantiate(Type<A>(), Type<B>())
							   .configure(grid, block, 0, cudaStream)
							   .launch(a, b, elems));
				#else
					jitifyCall(program.kernel("copyKernel")
							   .instantiate(Type<A>(), Type<B>())
							   .configure(grid, block)
							   .launch(a, b, elems));
				#endif // LIBRAPID_CUDA_STREAM
				}, dst.data, src.data);
			}
		}
	#else
		else
		{
			throw std::runtime_error("CUDA support was not enabled, so data "
									 "cannot be copied to the GPU");
		}
	#endif
	}
}

#endif // LIBRAPID_AUTOCAST