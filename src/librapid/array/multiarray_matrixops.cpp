#include <librapid/array/multiarray.hpp>
#include <librapid/utils/array_utils.hpp>
#include <librapid/autocast/autocast.hpp>

namespace librapid
{
	Array Array::dot(const Array& other) const {
        // Find the largest datatype and location
        Datatype resDtype = std::max(m_dtype, other.m_dtype);
        Accelerator resLocn = std::max(m_location, other.m_location);

        return {};
	}
}