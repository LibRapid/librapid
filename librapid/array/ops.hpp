#ifndef LIBRAPID_OPS
#define LIBRAPID_OPS

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>

namespace librapid
{
	namespace ops
	{
		const auto add = [](auto a, auto b)
		{
			return a + b;
		};
	}
}

#endif // LIBRAPID_OPS