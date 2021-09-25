#ifndef LIBRAPID_AUTOCAST
#define LIBRAPID_AUTOCAST

#include <librapid/autocast/custom_complex.hpp>

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
		INT8,			// int8_t
		UINT8,			// uint8_t
		INT16,			// int16_t
		UINT16,			// uint16_t
		INT32,			// int32_t
		UINT32,			// uint32_t
		INT64,			// int64_t
		UINT64,			// uint64_t
		FLOAT32,		// float
		FLOAT64,		// double
		CFLOAT32,		// librapid::Complex<float>
		CFLOAT64,		// librapid::Complex<double>
	};

	enum class Accelerator
	{
		NONE,
		CPU,
		GPU
	};

	/**
	 * \rst
	 *
	 * Contains a pointer (void pointer) to a memory
	 * location, as well as a datatype for that memory
	 *
	 * \endrst
	 */
	struct VoidPtr
	{
		void *ptr = nullptr;
		Datatype dtype = Datatype::NONE;
		Accelerator location = Accelerator::CPU;
	};

	extern VoidPtr validVoidPtr;

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
			case Datatype::INT8:
				return true;
			case Datatype::UINT8:
				return true;
			case Datatype::INT16:
				return true;
			case Datatype::UINT16:
				return true;
			case Datatype::INT32:
				return true;
			case Datatype::UINT32:
				return true;
			case Datatype::INT64:
				return true;
			case Datatype::UINT64:
				return true;
			default:
				return false;
		}
	}

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
			case Datatype::INT8:
				return false;
			case Datatype::UINT8:
				return true;
			case Datatype::INT16:
				return false;
			case Datatype::UINT16:
				return true;
			case Datatype::INT32:
				return false;
			case Datatype::UINT32:
				return true;
			case Datatype::INT64:
				return false;
			case Datatype::UINT64:
				return true;
			default:
				return false;
		}

		return false;
	}

	inline bool isFloating(Datatype t)
	{
		switch (t)
		{
			case Datatype::FLOAT32:
				return true;
			case Datatype::FLOAT64:
				return true;
			case Datatype::CFLOAT32:
				return true;
			case Datatype::CFLOAT64:
				return true;
			default:
				return false;
		}
	}

	inline bool isComplex(Datatype t)
	{
		switch (t)
		{
			case Datatype::CFLOAT32:
				return true;
			case Datatype::CFLOAT64:
				return true;
			default:
				return false;
		}
	}

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
			case Datatype::INT8:
				return sizeof(int8_t);
			case Datatype::UINT8:
				return sizeof(uint8_t);
			case Datatype::INT16:
				return sizeof(int16_t);
			case Datatype::UINT16:
				return sizeof(uint16_t);
			case Datatype::INT32:
				return sizeof(int32_t);
			case Datatype::UINT32:
				return sizeof(uint32_t);
			case Datatype::INT64:
				return sizeof(int64_t);
			case Datatype::UINT64:
				return sizeof(uint64_t);
			case Datatype::FLOAT32:
				return sizeof(float);
			case Datatype::FLOAT64:
				return sizeof(double);
			case Datatype::CFLOAT32:
				return sizeof(librapid::Complex<float>);
			case Datatype::CFLOAT64:
				return sizeof(librapid::Complex<double>);
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
			case Datatype::INT8:
				return "INT8";
			case Datatype::UINT8:
				return "UINT8";
			case Datatype::INT16:
				return "INT16";
			case Datatype::UINT16:
				return "UINT16";
			case Datatype::INT32:
				return "INT32";
			case Datatype::UINT32:
				return "UINT32";
			case Datatype::INT64:
				return "INT64";
			case Datatype::UINT64:
				return "UINT64";
			case Datatype::FLOAT32:
				return "FLOAT32";
			case Datatype::FLOAT64:
				return "FLOAT64";
			case Datatype::CFLOAT32:
				return "CFLOAT32";
			case Datatype::CFLOAT64:
				return "CFLOAT64";
		}

		return "UNKNOWN";
	}

	template<typename T>
	inline Datatype typeToDatatype(T x)
	{
		if (std::is_same<T, bool>) return Datatype::BOOL;
		if (std::is_same<T, int8_t>) return Datatype::INT8;
		if (std::is_same<T, uint8_t>) return Datatype::UINT8;
		if (std::is_same<T, int16_t>) return Datatype::INT16;
		if (std::is_same<T, uint16_t>) return Datatype::UINT16;
		if (std::is_same<T, int32_t>) return Datatype::INT32;
		if (std::is_same<T, uint32_t>) return Datatype::UINT32;
		if (std::is_same<T, int64_t>) return Datatype::INT64;
		if (std::is_same<T, uint64_t>) return Datatype::UINT64;
		if (std::is_same<T, float>) return Datatype::FLOAT32;
		if (std::is_same<T, double>) return Datatype::FLOAT64;
		if (std::is_same<T, Complex<float>>) return Datatype::CFLOAT32;
		if (std::is_same<T, Complex<double>>) return Datatype::CFLOAT64;

		return Datatype::NONE;
	}

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
			{Datatype::INT8, int8Str},
			{Datatype::UINT8, uint8Str},
			{Datatype::INT16, int16Str},
			{Datatype::UINT16, uint16Str},
			{Datatype::INT32, int32Str},
			{Datatype::UINT32, uint32Str},
			{Datatype::INT64, int64Str},
			{Datatype::UINT64, uint64Str},
			{Datatype::FLOAT32, float32Str},
			{Datatype::FLOAT64, float64Str},
			{Datatype::CFLOAT32, cfloat32Str},
			{Datatype::CFLOAT64, cfloat64Str}
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

	inline Accelerator stringToAccelerator(const std::string &str)
	{
		// Force the string to be lower case
		std::string temp = str;
		std::transform(temp.begin(), temp.end(), temp.begin(),
					   [](unsigned char c)
		{
			return std::tolower(c);
		});

		if (temp == "cpu")
			return Accelerator::CPU;

	#ifdef LIBRAPID_HAS_CUDA
		if (temp == "gpu")
			return Accelerator::GPU;
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

	template<typename T>
	inline void *AUTOCAST_ALLOC_(Accelerator locn, size_t elems)
	{
		if (locn == Accelerator::CPU)
			return alignedMalloc(sizeof(T) * elems);
	#ifdef LIBRAPID_HAS_CUDA
		void *res = nullptr;

	#ifdef LIBRAPID_CUDA_STREAM
		cudaSafeCall(cudaMallocAsync(&res, sizeof(T) * elems, cudaStream));
	#else
		cudaSafeCall(cudaMalloc(&res, sizeof(T) * elems));
	#endif

		return res;
	#else
		throw std::runtime_error("CUDA support was not enabled, so memory cannot "
								 "be allocated on the GPU");
	#endif
	}

	inline VoidPtr AUTOCAST_ALLOC(Datatype t, Accelerator locn, size_t elems)
	{
		switch (t)
		{
			case librapid::Datatype::NONE:
				{
					return librapid::VoidPtr{};
				}
			case librapid::Datatype::BOOL:
				{
					return {AUTOCAST_ALLOC_<bool>(locn, elems), t, locn};
				}
			case librapid::Datatype::INT8:
				{
					return {AUTOCAST_ALLOC_<int8_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::UINT8:
				{
					return {AUTOCAST_ALLOC_<uint8_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::INT16:
				{
					return {AUTOCAST_ALLOC_<int16_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::UINT16:
				{
					return {AUTOCAST_ALLOC_<uint16_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::INT32:
				{
					return {AUTOCAST_ALLOC_<int32_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::UINT32:
				{
					return {AUTOCAST_ALLOC_<uint32_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::INT64:
				{
					return {AUTOCAST_ALLOC_<int64_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::UINT64:
				{
					return {AUTOCAST_ALLOC_<uint64_t>(locn, elems), t, locn};
				}
			case librapid::Datatype::FLOAT32:
				{
					return {AUTOCAST_ALLOC_<float>(locn, elems), t, locn};
				}
			case librapid::Datatype::FLOAT64:
				{
					return {AUTOCAST_ALLOC_<double>(locn, elems), t, locn};
				}
			case librapid::Datatype::CFLOAT32:
				{
					return {AUTOCAST_ALLOC_<librapid::Complex<float>>(locn, elems),
						t, locn};
				}
			case librapid::Datatype::CFLOAT64:
				{
					return {AUTOCAST_ALLOC_<librapid::Complex<double>>(locn, elems),
						t, locn};
				}
		}

		return VoidPtr{};
	}

	inline void AUTOCAST_FREE(VoidPtr data)
	{
		if (data.location == Accelerator::CPU)
			alignedFree(data.ptr);
	#ifdef LIBRAPID_HAS_CUDA
		else if (data.location == Accelerator::GPU)
		{
		#ifdef LIBRAPID_CUDA_STREAM
			cudaSafeCall(cudaFreeAsync(data.ptr, cudaStream));
		#else
			cudaSafeCall(cudaFree(data.ptr));
		#endif
		}
	#else
		throw std::runtime_error("CUDA support was not enabled, so device memory cannot be freed");
	#endif
	}

#define AUTOCAST_UNARY_(f, locnA, precast, res, ...)												\
			switch (res.dtype)																				\
			{																								\
				case librapid::Datatype::NONE:																\
				{																							\
					throw std::invalid_argument("Cannot run function on NONETYPE");							\
					break;																					\
				}																							\
				case librapid::Datatype::VALIDNONE:															\
				{																							\
					f(locnA, res.location, precast, (bool *) nullptr, __VA_ARGS__);							\
					break;																					\
				}																							\
				case librapid::Datatype::BOOL:																\
				{																							\
					f(locnA, res.location, precast, (bool *) res.ptr, __VA_ARGS__);							\
					break;																					\
				}																							\
				case librapid::Datatype::INT8:																\
				{																							\
					f(locnA, res.location, precast, (int8_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::UINT8:																\
				{																							\
					f(locnA, res.location, precast, (uint8_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::INT16:																\
				{																							\
					f(locnA, res.location, precast, (int16_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::UINT16:															\
				{																							\
					f(locnA, res.location, precast, (uint16_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::INT32:																\
				{																							\
					f(locnA, res.location, precast, (int32_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::UINT32:															\
				{																							\
					f(locnA, res.location, precast, (uint32_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::INT64:																\
				{																							\
					f(locnA, res.location, precast, (int64_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::UINT64:															\
				{																							\
					f(locnA, res.location, precast, (uint64_t *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::FLOAT32:															\
				{																							\
					f(locnA, res.location, precast, (float *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::FLOAT64:															\
				{																							\
					f(locnA, res.location, precast, (double *) res.ptr, __VA_ARGS__);						\
					break;																					\
				}																							\
				case librapid::Datatype::CFLOAT32:															\
				{																							\
					f(locnA, res.location, precast, (librapid::Complex<float> *) res.ptr, __VA_ARGS__);		\
					break;																					\
				}																							\
				case librapid::Datatype::CFLOAT64:															\
				{																							\
					f(locnA, res.location, precast, (librapid::Complex<double> *) res.ptr, __VA_ARGS__);	\
					break;																					\
				}																							\
			}

#define AUTOCAST_UNARY(f, vptr, res, ...)																\
			switch (vptr.dtype)																					\
			{																									\
				case librapid::Datatype::NONE:																	\
				{																								\
					throw std::invalid_argument("Cannot run function on NONETYPE");								\
					break;																						\
				}																								\
				case librapid::Datatype::VALIDNONE:																\
				{																								\
					break;																						\
				}																								\
				case librapid::Datatype::BOOL:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (bool *) vptr.ptr, res, __VA_ARGS__)						\
					break;																						\
				}																								\
				case librapid::Datatype::INT8:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (int8_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::UINT8:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (uint8_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::INT16:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (int16_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::UINT16:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (uint16_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::INT32:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (int32_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::UINT32:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (uint32_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::INT64:																	\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (int64_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::UINT64:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (uint64_t *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::FLOAT32:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (float *) vptr.ptr, res, __VA_ARGS__)						\
					break;																						\
				}																								\
				case librapid::Datatype::FLOAT64:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (double *) vptr.ptr, res, __VA_ARGS__)					\
					break;																						\
				}																								\
				case librapid::Datatype::CFLOAT32:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (librapid::Complex<float> *) vptr.ptr, res, __VA_ARGS__)	\
					break;																						\
				}																								\
				case librapid::Datatype::CFLOAT64:																\
				{																								\
					AUTOCAST_UNARY_(f, vptr.location, (librapid::Complex<double> *) vptr.ptr, res, __VA_ARGS__)	\
					break;																						\
				}																								\
			}

#define AUTOCAST_BINARY__(f, locnA, locnB, precastA, precastB, res, ...)											\
	switch (res.dtype)																								\
	{																												\
		case librapid::Datatype::NONE:																				\
		{																											\
			throw std::invalid_argument("Cannot run function on NONETYPE");											\
		}																											\
		case librapid::Datatype::BOOL:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (bool *) res.ptr, __VA_ARGS__);						\
			break;																									\
		}																											\
		case librapid::Datatype::INT8:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (int8_t *) res.ptr, __VA_ARGS__);						\
			break;																									\
		}																											\
		case librapid::Datatype::UINT8:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (uint8_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::INT16:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (int16_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::UINT16:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (uint16_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::INT32:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (int32_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::UINT32:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (uint32_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::INT64:																				\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (int64_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::UINT64:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (uint64_t *) res.ptr, __VA_ARGS__);					\
			break;																									\
		}																											\
		case librapid::Datatype::FLOAT32:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (float *) res.ptr, __VA_ARGS__);						\
			break;																									\
		}																											\
		case librapid::Datatype::FLOAT64:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (double *) res.ptr, __VA_ARGS__);						\
			break;																									\
		}																											\
		case librapid::Datatype::CFLOAT32:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (librapid::Complex<float> *) res.ptr, __VA_ARGS__);	\
			break;																									\
		}																											\
		case librapid::Datatype::CFLOAT64:																			\
		{																											\
			f(locnA, locnB, res.location, precastA, precastB, (librapid::Complex<double> *) res.ptr, __VA_ARGS__);	\
			break;																									\
		}																											\
	}

#define AUTOCAST_BINARY_(f, locnA, precastA, vptrB, res, ...)													\
	switch (vptrB.dtype)																				\
	{																									\
		case librapid::Datatype::NONE:																	\
		{																								\
			throw std::invalid_argument("Cannot run function on NONETYPE");								\
		}																								\
		case librapid::Datatype::BOOL:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (bool *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::INT8:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (int8_t *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::UINT8:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (uint8_t *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT16:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (int16_t *) vptrB.ptr, res, __VA_ARGS__)							\
			break;																						\
		}																								\
		case librapid::Datatype::UINT16:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (uint16_t *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT32:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (int32_t *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::UINT32:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (uint32_t *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT64:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (int64_t *) vptrB.ptr, res, __VA_ARGS__)					\
			break;																						\
		}																								\
		case librapid::Datatype::UINT64:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (uint64_t *) vptrB.ptr, res, __VA_ARGS__)			\
			break;																						\
		}																								\
		case librapid::Datatype::FLOAT32:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (float *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::FLOAT64:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (double *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::CFLOAT32:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (librapid::Complex<float> *) vptrB.ptr, res, __VA_ARGS__)	\
			break;																						\
		}																								\
		case librapid::Datatype::CFLOAT64:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (librapid::Complex<double> *) vptrB.ptr, res, __VA_ARGS__)	\
			break;																						\
		}																								\
	}

#define AUTOCAST_BINARY(f, vptrA, vptrB, res, ...)													\
	switch (vptrA.dtype)																			\
	{																								\
		case librapid::Datatype::NONE:																\
		{																							\
			throw std::invalid_argument("Cannot run function on NONETYPE");							\
		}																							\
		case librapid::Datatype::BOOL:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (bool *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::INT8:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (int8_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::UINT8:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (uint8_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT16:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (int16_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)							\
			break;																					\
		}																							\
		case librapid::Datatype::UINT16:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (uint16_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT32:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (int32_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::UINT32:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (uint32_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT64:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (int64_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)					\
			break;																					\
		}																							\
		case librapid::Datatype::UINT64:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (uint64_t *) vptrA.ptr, vptrB, res, __VA_ARGS__)			\
			break;																					\
		}																							\
		case librapid::Datatype::FLOAT32:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (float *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::FLOAT64:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (double *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::CFLOAT32:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (librapid::Complex<float> *) vptrA.ptr, vptrB, res, __VA_ARGS__)	\
			break;																					\
		}																							\
		case librapid::Datatype::CFLOAT64:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (librapid::Complex<double> *) vptrA.ptr, vptrB, res, __VA_ARGS__)	\
			break;																					\
		}																							\
	}

	namespace imp
	{
		template<typename A, typename B>
		inline void cpyCPU(const librapid::Accelerator &locnA,
						   const librapid::Accelerator &locnB,
						   A *dst, B *src, size_t size)
		{
			if (size < 250 * 250)
			{
				for (size_t i = 0; i < size; ++i)
				{
					dst[i] = (A) src[i];
				}
			}
			else
			{
			#pragma omp parallel for shared(dst, src, size)
				for (int64_t i = 0; i < (int64_t) size; ++i)
				{
					dst[i] = (A) src[i];
				}
			}
		}
	}

	namespace imp
	{
		template<typename A, typename B>
		inline void autocastMemcpyHelper(Accelerator locnA, Accelerator locnB,
										 A *__restrict a, B *__restrict b,
										 size_t elems)
		{
			if (locnA == Accelerator::CPU && locnB == Accelerator::CPU)
			{
				for (size_t i = 0; i < elems; ++i)
					a[i] = (A) b[i];
			}
		#ifdef LIBRAPID_HAS_CUDA
			else
			{
				if (locnB == Accelerator::CPU)
				{
					// Copy from CPU to GPU

					A newVal;

					for (size_t i = 0; i < elems; ++i)
					{
						newVal = (A) b[i];

					#ifdef LIBRAPID_CUDA_STREAM
						cudaSafeCall(cudaMemcpyAsync(a + i, &newVal, sizeof(A),
									 cudaMemcpyHostToDevice, cudaStream));
					#else
						cudaSafeCall(cudaMemcpy(a + i, &newVal, sizeof(A),
									 cudaMemcpyHostToDevice));
					#endif
					}
				}
				else
				{
					if (locnA == Accelerator::CPU)
					{
						// Copy from GPU to CPU

						B newVal = 0;

						for (size_t i = 0; i < elems; ++i)
						{
						#ifdef LIBRAPID_CUDA_STREAM
							cudaSafeCall(cudaMemcpyAsync(&newVal, b + i, sizeof(A),
										 cudaMemcpyDeviceToHost, cudaStream));
						#else
							cudaSafeCall(cudaMemcpy(&newVal, b + i, sizeof(A),
										 cudaMemcpyDeviceToHost));
						#endif

							a[i] = (A) newVal;
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

						// Use 1 to 256 threads per block
						if (elems < 256)
						{
							threadsPerBlock = (uint16_t) elems;
							blocksPerGrid = 1;
						}
						else
						{
							threadsPerBlock = 256;
							blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
						}

						dim3 grid(blocksPerGrid);
						dim3 block(threadsPerBlock);

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
					}
				}
			}
		#endif
		}
	}

	inline void AUTOCAST_MEMCPY(VoidPtr dst, const VoidPtr &src, size_t elems)
	{
		if (src.location == Accelerator::CPU && dst.location == Accelerator::CPU)
		{
			if (src.dtype == dst.dtype)
				memcpy(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems);
			else
				AUTOCAST_UNARY(imp::cpyCPU, dst, src, elems);
		}
	#ifdef LIBRAPID_HAS_CUDA
		else
		{
			if (src.dtype == dst.dtype)
			{
			#ifdef LIBRAPID_CUDA_STREAM
				if (src.location == Accelerator::CPU && dst.location == Accelerator::GPU)
					cudaSafeCall(cudaMemcpyAsync(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyHostToDevice, cudaStream));
				else if (src.location == Accelerator::GPU && dst.location == Accelerator::CPU)
					cudaSafeCall(cudaMemcpyAsync(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyDeviceToHost, cudaStream));
				else if (src.location == Accelerator::GPU && dst.location == Accelerator::GPU)
					cudaSafeCall(cudaMemcpyAsync(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyDeviceToDevice, cudaStream));
			#else
				if (src.location == Accelerator::CPU && dst.location == Accelerator::GPU)
					cudaSafeCall(cudaMemcpy(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyHostToDevice));
				else if (src.location == Accelerator::GPU && dst.location == Accelerator::CPU)
					cudaSafeCall(cudaMemcpy(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyDeviceToHost));
				else if (src.location == Accelerator::GPU && dst.location == Accelerator::GPU)
					cudaSafeCall(cudaMemcpy(dst.ptr, src.ptr, datatypeBytes(src.dtype) * elems, cudaMemcpyDeviceToDevice));
			#endif
			}
			else
			{
				// Datatypes are not the same, so memcpy will not work
				AUTOCAST_UNARY(imp::autocastMemcpyHelper, dst, src, elems);
			}
		}
	#endif
	}
}

#endif // LIBRAPID_AUTOCAST