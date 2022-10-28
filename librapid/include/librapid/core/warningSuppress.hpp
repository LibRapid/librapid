#ifndef LIBRAPID_WARNING_SUPPRESS
#define LIBRAPID_WARNING_SUPPRESS

#ifdef _MSC_VER
#	define LIBRAPID_MSVC_SUPPRESS(WARNING_) __pragma(warning(suppress : WARNING_))
#else
#	define LIBRAPID_MSVC_SUPPRESS(WARNING_)
#endif

// Disable warnings for GCC/Clang
#ifdef __GNUC__
#	define LIBRAPID_GCC_SUPPRESS(WARNING_)                                                        \
		_Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-W" #WARNING_ "\"")
#else
#	define LIBRAPID_GCC_SUPPRESS(WARNING_)
#endif

LIBRAPID_MSVC_SUPPRESS(4996) // Disable warnings about unsafe classes
LIBRAPID_MSVC_SUPPRESS(4723) // Disable zero division errors
LIBRAPID_MSVC_SUPPRESS(5245) // unreferenced function with internal linkage has been removed

#endif // LIBRAPID_WARNING_SUPPRESS