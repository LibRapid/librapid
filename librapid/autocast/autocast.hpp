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
		BOOL,			// bool
		CHAR,			// char
		UCHAR,			// unsigned char
		INT16,			// int
		UINT16,			// unsigned int
		INT32,			// long
		UINT32,			// unsigned long
		INT64,			// long long
		UINT64,			// unsigned long long
		FLOAT32,		// float
		FLOAT64,		// double
		CFLOAT32,		// librapid::complex<float>
		CFLOAT64,		// librapid::complex<double>
	};

	enum class Accelerator
	{
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
		Accelerator location;
	};

	size_t datatype_bytes(Datatype t)
	{
		switch (t)
		{
			case Datatype::NONE:
				return 0;
			case Datatype::BOOL:
				return sizeof(bool);
			case Datatype::CHAR:
				return sizeof(char);
			case Datatype::UCHAR:
				return sizeof(unsigned char);
			case Datatype::INT16:
				return sizeof(int);
			case Datatype::UINT16:
				return sizeof(unsigned int);
			case Datatype::INT32:
				return sizeof(long);
			case Datatype::UINT32:
				return sizeof(unsigned long);
			case Datatype::INT64:
				return sizeof(long long);
			case Datatype::UINT64:
				return sizeof(unsigned long long);
			case Datatype::FLOAT32:
				return sizeof(float);
			case Datatype::FLOAT64:
				return sizeof(double);
			case Datatype::CFLOAT32:
				return sizeof(librapid::complex<float>);
			case Datatype::CFLOAT64:
				return sizeof(librapid::complex<double>);
		}

		return 0;
	}

	template<typename T>
	LR_INLINE void *AUTOCAST_ALLOC_(Accelerator locn, size_t elems)
	{
		if (locn == Accelerator::CPU)
			return malloc(sizeof(T) * elems);
		throw std::bad_alloc();
	}

	LR_INLINE VoidPtr AUTOCAST_ALLOC(Datatype t, Accelerator locn, size_t elems)
	{
		switch (t)
		{
			case librapid::Datatype::NONE:
				{
					return librapid::VoidPtr{};
				}
			case librapid::Datatype::BOOL:
				{
					return {AUTOCAST_ALLOC_<bool>(locn, elems), t};
				}
			case librapid::Datatype::CHAR:
				{
					return {AUTOCAST_ALLOC_<char>(locn, elems), t};
				}
			case librapid::Datatype::UCHAR:
				{
					return {AUTOCAST_ALLOC_<unsigned char>(locn, elems), t};
				}
			case librapid::Datatype::INT16:
				{
					return {AUTOCAST_ALLOC_<int>(locn, elems), t};
				}
			case librapid::Datatype::UINT16:
				{
					return {AUTOCAST_ALLOC_<unsigned int>(locn, elems), t};
				}
			case librapid::Datatype::INT32:
				{
					return {AUTOCAST_ALLOC_<long>(locn, elems), t};
				}
			case librapid::Datatype::UINT32:
				{
					return {AUTOCAST_ALLOC_<unsigned long>(locn, elems), t};
				}
			case librapid::Datatype::INT64:
				{
					return {AUTOCAST_ALLOC_<long long>(locn, elems), t};
				}
			case librapid::Datatype::UINT64:
				{
					return {AUTOCAST_ALLOC_<unsigned long long>(locn, elems), t};
				}
			case librapid::Datatype::FLOAT32:
				{
					return {AUTOCAST_ALLOC_<float>(locn, elems), t};
				}
			case librapid::Datatype::FLOAT64:
				{
					return {AUTOCAST_ALLOC_<double>(locn, elems), t};
				}
			case librapid::Datatype::CFLOAT32:
				{
					return {AUTOCAST_ALLOC_<librapid::complex<float>>(locn, elems), t};
				}
			case librapid::Datatype::CFLOAT64:
				{
					return {AUTOCAST_ALLOC_<librapid::complex<double>>(locn, elems), t};
				}
		}

		return VoidPtr{};
	}

	LR_INLINE void AUTOCAST_FREE(VoidPtr data)
	{
		free(data.ptr);
	}

	LR_INLINE void AUTOCAST_FREE(void *data)
	{
		free(data);
	}

#define AUTOCAST_UNARY_(f, locnA, precast, res, ...)									\
	switch (res.dtype)															\
	{																			\
		case librapid::Datatype::NONE:											\
		{																		\
			throw std::invalid_argument("Cannot run function on NONETYPE");		\
		}																		\
		case librapid::Datatype::BOOL:											\
		{																		\
			f(locnA, res.location, precast, (bool *) res.ptr, __VA_ARGS__);							\
			break;																\
		}																		\
		case librapid::Datatype::CHAR:											\
		{																		\
			f(locnA, res.location, precast, (char *) res.ptr, __VA_ARGS__);							\
			break;																\
		}																		\
		case librapid::Datatype::UCHAR:											\
		{																		\
			f(locnA, res.location, precast, (unsigned char *) res.ptr, __VA_ARGS__);					\
			break;																\
		}																		\
		case librapid::Datatype::INT16:											\
		{																		\
			f(locnA, res.location, precast, (int *) res.ptr, __VA_ARGS__);							\
			break;																\
		}																		\
		case librapid::Datatype::UINT16:										\
		{																		\
			f(locnA, res.location, precast, (unsigned int *) res.ptr, __VA_ARGS__);					\
			break;																\
		}																		\
		case librapid::Datatype::INT32:											\
		{																		\
			f(locnA, res.location, precast, (long *) res.ptr, __VA_ARGS__);							\
			break;																\
		}																		\
		case librapid::Datatype::UINT32:										\
		{																		\
			f(locnA, res.location, precast, (unsigned long *) res.ptr, __VA_ARGS__);					\
			break;																\
		}																		\
		case librapid::Datatype::INT64:											\
		{																		\
			f(locnA, res.location, precast, (long long *) res.ptr, __VA_ARGS__);						\
			break;																\
		}																		\
		case librapid::Datatype::UINT64:										\
		{																		\
			f(locnA, res.location, precast, (unsigned long long *) res.ptr, __VA_ARGS__);			\
			break;																\
		}																		\
		case librapid::Datatype::FLOAT32:										\
		{																		\
			f(locnA, res.location, precast, (float *) res.ptr, __VA_ARGS__);							\
			break;																\
		}																		\
		case librapid::Datatype::FLOAT64:										\
		{																		\
			f(locnA, res.location, precast, (double *) res.ptr, __VA_ARGS__);						\
			break;																\
		}																		\
		case librapid::Datatype::CFLOAT32:										\
		{																		\
			f(locnA, res.location, precast, (librapid::complex<float> *) res.ptr, __VA_ARGS__);		\
			break;																\
		}																		\
		case librapid::Datatype::CFLOAT64:										\
		{																		\
			f(locnA, res.location, precast, (librapid::complex<double> *) res.ptr, __VA_ARGS__);		\
			break;																\
		}																		\
	}

#define AUTOCAST_UNARY(f, vptr, res, ...)													\
	switch (vptr.dtype)																		\
	{																						\
		case librapid::Datatype::NONE:														\
		{																					\
			throw std::invalid_argument("Cannot run function on NONETYPE");					\
		}																					\
		case librapid::Datatype::BOOL:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (bool *) vptr.ptr, res, __VA_ARGS__)							\
			break;																			\
		}																					\
		case librapid::Datatype::CHAR:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (char *) vptr.ptr, res, __VA_ARGS__)							\
			break;																			\
		}																					\
		case librapid::Datatype::UCHAR:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (unsigned char *) vptr.ptr, res, __VA_ARGS__)				\
			break;																			\
		}																					\
		case librapid::Datatype::INT16:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (int *) vptr.ptr, res, __VA_ARGS__)							\
			break;																			\
		}																					\
		case librapid::Datatype::UINT16:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (unsigned int *) vptr.ptr, res, __VA_ARGS__)					\
			break;																			\
		}																					\
		case librapid::Datatype::INT32:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (long *) vptr.ptr, res, __VA_ARGS__)							\
			break;																			\
		}																					\
		case librapid::Datatype::UINT32:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (unsigned long *) vptr.ptr, res, __VA_ARGS__)				\
			break;																			\
		}																					\
		case librapid::Datatype::INT64:														\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (long long *) vptr.ptr, res, __VA_ARGS__)					\
			break;																			\
		}																					\
		case librapid::Datatype::UINT64:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (unsigned long long *) vptr.ptr, res, __VA_ARGS__)			\
			break;																			\
		}																					\
		case librapid::Datatype::FLOAT32:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (float *) vptr.ptr, res, __VA_ARGS__)						\
			break;																			\
		}																					\
		case librapid::Datatype::FLOAT64:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (double *) vptr.ptr, res, __VA_ARGS__)						\
			break;																			\
		}																					\
		case librapid::Datatype::CFLOAT32:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (librapid::complex<float> *) vptr.ptr, res, __VA_ARGS__)		\
			break;																			\
		}																					\
		case librapid::Datatype::CFLOAT64:													\
		{																					\
			AUTOCAST_UNARY_(f, vptr.location, (librapid::complex<double> *) vptr.ptr, res, __VA_ARGS__)	\
			break;																			\
		}																					\
	}

#define AUTOCAST_BINARY__(f, locnA, locnB, precastA, precastB, res, ...)							\
	switch (res.dtype)															\
	{																				\
		case librapid::Datatype::NONE:												\
		{																			\
			throw std::invalid_argument("Cannot run function on NONETYPE");			\
		}																			\
		case librapid::Datatype::BOOL:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (bool *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::CHAR:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (char *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::UCHAR:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (unsigned char *) res.ptr, __VA_ARGS__);				\
			break;																	\
		}																			\
		case librapid::Datatype::INT16:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (int *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::UINT16:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (unsigned int *) res.ptr, __VA_ARGS__);				\
			break;																	\
		}																			\
		case librapid::Datatype::INT32:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (long *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::UINT32:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (unsigned long *) res.ptr, __VA_ARGS__);				\
			break;																	\
		}																			\
		case librapid::Datatype::INT64:												\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (long long *) res.ptr, __VA_ARGS__);					\
			break;																	\
		}																			\
		case librapid::Datatype::UINT64:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (unsigned long long *) res.ptr, __VA_ARGS__);			\
			break;																	\
		}																			\
		case librapid::Datatype::FLOAT32:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (float *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::FLOAT64:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (double *) res.ptr, __VA_ARGS__);						\
			break;																	\
		}																			\
		case librapid::Datatype::CFLOAT32:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (librapid::complex<float> *) res.ptr, __VA_ARGS__);	\
			break;																	\
		}																			\
		case librapid::Datatype::CFLOAT64:											\
		{																			\
			f(locnA, locnB, res.location, precastA, precastB, (librapid::complex<double> *) res.ptr, __VA_ARGS__);	\
			break;																	\
		}																			\
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
		case librapid::Datatype::CHAR:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (char *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::UCHAR:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (unsigned char *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT16:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (int *) vptrB.ptr, res, __VA_ARGS__)							\
			break;																						\
		}																								\
		case librapid::Datatype::UINT16:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (unsigned int *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT32:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (long *) vptrB.ptr, res, __VA_ARGS__)						\
			break;																						\
		}																								\
		case librapid::Datatype::UINT32:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (unsigned long *) vptrB.ptr, res, __VA_ARGS__)				\
			break;																						\
		}																								\
		case librapid::Datatype::INT64:																	\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (long long *) vptrB.ptr, res, __VA_ARGS__)					\
			break;																						\
		}																								\
		case librapid::Datatype::UINT64:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (unsigned long long *) vptrB.ptr, res, __VA_ARGS__)			\
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
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (librapid::complex<float> *) vptrB.ptr, res, __VA_ARGS__)	\
			break;																						\
		}																								\
		case librapid::Datatype::CFLOAT64:																\
		{																								\
			AUTOCAST_BINARY__(f, locnA, vptrB.location, precastA, (librapid::complex<double> *) vptrB.ptr, res, __VA_ARGS__)	\
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
		case librapid::Datatype::CHAR:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (char *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::UCHAR:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (unsigned char *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT16:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (int *) vptrA.ptr, vptrB, res, __VA_ARGS__)							\
			break;																					\
		}																							\
		case librapid::Datatype::UINT16:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (unsigned int *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT32:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (long *) vptrA.ptr, vptrB, res, __VA_ARGS__)						\
			break;																					\
		}																							\
		case librapid::Datatype::UINT32:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (unsigned long *) vptrA.ptr, vptrB, res, __VA_ARGS__)				\
			break;																					\
		}																							\
		case librapid::Datatype::INT64:																\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (long long *) vptrA.ptr, vptrB, res, __VA_ARGS__)					\
			break;																					\
		}																							\
		case librapid::Datatype::UINT64:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (unsigned long long *) vptrA.ptr, vptrB, res, __VA_ARGS__)			\
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
			AUTOCAST_BINARY_(f, vptrA.location, (librapid::complex<float> *) vptrA.ptr, vptrB, res, __VA_ARGS__)	\
			break;																					\
		}																							\
		case librapid::Datatype::CFLOAT64:															\
		{																							\
			AUTOCAST_BINARY_(f, vptrA.location, (librapid::complex<double> *) vptrA.ptr, vptrB, res, __VA_ARGS__)	\
			break;																					\
		}																							\
	}
}

#endif // LIBRAPID_AUTOCAST