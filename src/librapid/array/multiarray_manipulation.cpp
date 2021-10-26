#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid
{
	void Array::transpose(const Extent &order)
	{
		Extent tempOrder;

		if (order.ndim() == 0)
		{
			// Order was not provided, so initialize it to be n, n-1, ... , 2, 0
			tempOrder = Extent(m_extent.ndim());
			for (int64_t i = 0; i < m_extent.ndim(); ++i)
				tempOrder[i] = m_extent.ndim() - i - 1;
		}
		else
		{
			tempOrder = order;
		}

		// Check the dimensions are correct
		if (tempOrder.ndim() != m_extent.ndim())
		{
			throw std::invalid_argument(std::to_string(tempOrder.ndim()) + " indices "
										"were passed to Array transpose, though "
										+ std::to_string(m_extent.ndim()) +
										" indices are required");
		}

		bool valid = true;
		int64_t count = 0;
		int64_t missing[LIBRAPID_MAX_DIMS]{};
		for (int64_t i = 0; i < ndim(); ++i)
		{
			if (std::count(&(*tempOrder.begin()), &(*tempOrder.end()), i) != 1)
			{
				missing[count++] = i;
				valid = false;
			}
		}

		if (!valid)
		{
			auto stream = std::stringstream();
			for (int64_t i = 0; i < count; ++i)
			{
				if (i == m_stride.ndim() - 1) stream << missing[i];
				else stream << missing[i] << ", ";
			}
			std::string missing_str = "(" + stream.str() + ")";

			throw std::runtime_error("Transpose requires that each index is passed "
									 "exactly once, but indices " + missing_str +
									 " were passed more than once or not at all");
		}

		m_extent.reorder(tempOrder.toVec());
		m_stride.reorder(tempOrder.toVec());
	}
}