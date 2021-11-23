#ifndef NDARRAY_ERRORS
#define NDARRAY_ERRORS

namespace librapid {
	enum class errors {
		ALL_OK = 0,
		INDEX_OUT_OF_RANGE = 1,
		ARRAY_DIMENSIONS_TOO_LARGE = 2
	};
}

#endif // NDARRAY_ERRORS