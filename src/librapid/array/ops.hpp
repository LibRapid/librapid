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
		struct Add
		{
			template<typename A, typename B>
			LR_INLINE typename std::common_type<A, B>::type operator()(A a, B b) const
			{
				typedef typename std::common_type<A, B>::type common;
				return (common) a + (common) b;
			}
		};
	}
}

#endif // LIBRAPID_OPS