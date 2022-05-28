#ifndef LIBRAPID_INCLUDE
#define LIBRAPID_INCLUDE

#include "VERSION.hpp"
#include "internal/config.hpp"
#include "internal/memUtils.hpp"
#include "utils/time.hpp"
#include "utils/console.hpp"
#include "math/constants.hpp"
#include "math/coreMath.hpp"
#include "math/statistics.hpp"
#include "math/advanced.hpp"

// Disable zero-division warnings for the vector library
#pragma warning(push)
#pragma warning(disable : 4723)
#include "math/vector.hpp"
#pragma warning(pop)

#include "internal/forward.hpp"
#include "array/traits.hpp"
#include "array/denseStorage.hpp"
#include "array/helpers/extent.hpp"
#include "array/arrayBase.hpp"
#include "array/cwisebinop.hpp"
#include "array/cwiseunop.hpp"
#include "array/array.hpp"

#endif // LIBRAPID_INCLUDE