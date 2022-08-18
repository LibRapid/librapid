#ifndef LIBRAPID_INCLUDE
#define LIBRAPID_INCLUDE

#pragma warning(push)
#pragma warning(disable : 4723)
#pragma warning(disable : 4804)
#pragma warning(disable : 4146) // Unary minus applied to unsigned type
#pragma warning(disable : 4018) // Signed/unsigned mismatch
#pragma warning(disable : 4244) // Possible loss of data in conversion
#pragma warning(disable : 4267) // Possible loss of data in conversion

#define MPIRXX_HAVE_LLONG // Enable long long support

#include "VERSION.hpp"
#include "internal/config.hpp"
#include "linalg/threadHelper.hpp"
#include "modified/modified.hpp"
#include "cuda/cudaCodeLoader.hpp"
#include "internal/memUtils.hpp"
#include "utils/traits.hpp"
#include "utils/time.hpp"
#include "utils/console.hpp"
#include "utils/bit.hpp"
#include "librapid/linalg/blasInterface.hpp"
#include "math/constants.hpp"
#include "math/coreMath.hpp"
#include "math/statistics.hpp"
#include "math/advanced.hpp"
#include "math/fastMath.hpp"
#include "math/zTheory.hpp"
#include "math/vector.hpp"
#include "math/complex.hpp"
#include "internal/forward.hpp"
#include "array/helpers/kernelHelper.hpp"
#include "array/denseStorage.hpp"
#include "array/helpers/extent.hpp"
#include "array/arrayBase.hpp"
#include "array/cwisebinop.hpp"
#include "array/cwiseunop.hpp"
#include "array/array.hpp"
#include "test/test.hpp"

#include "utils/toString.hpp"

#pragma warning(pop)

#endif // LIBRAPID_INCLUDE