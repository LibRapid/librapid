#ifndef LIBRAPID_INCLUDE
#define LIBRAPID_INCLUDE

#pragma warning(push)
// Disable zero-division warnings for the vector library
#pragma warning(disable : 4723)

// Disable zero-division warnings for the vector library
#pragma warning(disable : 4804)

#include "VERSION.hpp"
#include "internal/config.hpp"
#include "linalg/threadHelper.hpp"
#include "modified/modified.hpp"
#include "cuda/cudaCodeLoader.hpp"
#include "internal/memUtils.hpp"
#include "utils/time.hpp"
#include "utils/console.hpp"
#include "librapid/linalg/blasInterface.hpp"
#include "math/constants.hpp"
#include "math/coreMath.hpp"
#include "math/statistics.hpp"
#include "math/advanced.hpp"
#include "math/fastMath.hpp"
#include "math/zTheory.hpp"
#include "math/vector.hpp"
#include "internal/forward.hpp"
#include "array/traits.hpp"
#include "array/helpers/kernelHelper.hpp"
#include "array/denseStorage.hpp"
#include "array/helpers/extent.hpp"
#include "array/arrayBase.hpp"
#include "array/cwisebinop.hpp"
#include "array/cwiseunop.hpp"
#include "array/array.hpp"

#pragma warning(pop)

#endif // LIBRAPID_INCLUDE