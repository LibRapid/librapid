#ifndef LIBRAPID_INCLUDE
#define LIBRAPID_INCLUDE

#pragma warning(push)
#pragma warning(disable : 4723)
#pragma warning(disable : 4804)
#pragma warning(disable : 4146) // Unary minus applied to unsigned type
#pragma warning(disable : 4018) // Signed/unsigned mismatch
#pragma warning(disable : 4244) // Possible loss of data in conversion
#pragma warning(disable : 4267) // Possible loss of data in conversion

#define MPIRXX_HAVE_LLONG // Enable i64 support

#include "VERSION.hpp"
#include "internal/config.hpp"

#if defined(LIBRAPID_USE_MULTIPREC)
// MPIR (modified) for BigNumber types
#	include <mpirxx.h>
#	include <mpreal.h>
#endif

#include "internal/typedefs.hpp"
#include "internal/forward.hpp"
#include "math/mpfr.hpp"

#include "utils/traits.hpp"
#include "utils/time.hpp"
#include "utils/console.hpp"
#include "utils/bit.hpp"

#include "math/constants.hpp"
#include "math/coreMath.hpp"
#include "math/vector.hpp"
#include "math/fastMath.hpp"
#include "math/zTheory.hpp"
#include "math/complex.hpp"
#include "math/advanced.hpp"
#include "math/statistics.hpp"

#include "modified/modified.hpp"
#include "cuda/cudaCodeLoader.hpp"
#include "internal/memUtils.hpp"
#include "cuda/memUtils.hpp"
#include "linalg/threadHelper.hpp"
#include "linalg/blasInterface.hpp"

#include "array/denseStorage.hpp"
#include "array/valueReference.hpp"
#include "array/helpers/extent.hpp"
#include "array/helpers/kernelFormat.hpp"
#include "array/functors/functors.hpp"
#include "array/cwisebinop.hpp"
#include "array/cwiseunop.hpp"
#include "array/cwisemap.hpp"
#include "array/cast.hpp"
#include "array/commaInitializer.hpp"
#include "array/arrayBase.hpp"
#include "array/array.hpp"

#include "utils/suffix.hpp"
#include "test/test.hpp"
#include "utils/toString.hpp"

#pragma warning(pop)

#endif // LIBRAPID_INCLUDE