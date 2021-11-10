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

	Array concatenate(const std::vector<Array> &arrays, int64_t axis)
	{
		int64_t dims = arrays[0].ndim();
		if (axis < 0 || axis > dims)
			throw std::range_error("Index " + std::to_string(axis)
								   + " is out of range for array with "
								   + std::to_string(dims) + " dimensions");

		if (arrays.empty()) return Array();
		if (arrays.size() == 1) return arrays[0].clone();

		// Check all arrays are the same size and compute the new dimension for the
		// resulting array
		int64_t newDim = arrays[0].extent()[axis];
		const Extent &dim0 = arrays[0].extent();

		Datatype resDtype = arrays[0].dtype();
		Accelerator resLocn = arrays[0].location();

		for (uint64_t i = 1; i < arrays.size(); ++i)
		{
			if (arrays[i].isScalar())
				throw std::invalid_argument("Cannot concatenate scalar values");

			if (arrays[i].ndim() != dims)
				throw std::invalid_argument("To concatenate arrays, all "
											"values must have the same number of "
											"dimensions, however (at least) one "
											"array failed this condition. "
											+ std::to_string(arrays[i].ndim())
											+ " dimensions is not compatible with "
											+ std::to_string(dims) + " dimensions");

			// Ensure every dimension other than <axis> is equal
			// (concatenating on <axis> allows for different size in that dimension)
			for (int64_t j = 0; j < dims; ++j)
			{
				if (j != axis && arrays[i].extent()[j] != dim0[j])
					throw std::invalid_argument("To concatenate arrays, all "
												"dimensions other than index <axis> "
												"must be equal, however (at least) "
												"one array failed this condition. "
												+ arrays[i].extent().str()
												+ " is not compatible with "
												+ dim0.str());
			}

			newDim += arrays[i].extent()[axis];
			if (arrays[i].dtype() > resDtype) resDtype = arrays[i].dtype();
			if (arrays[i].location() > resLocn) resLocn = arrays[i].location();
		}

		int64_t step = 1;
		for (int64_t d = axis; d < dims; ++d)
			step *= dim0[d];

		Extent resShape(arrays[0].extent());
		resShape[axis] = newDim;

		Array res(resShape, resDtype, resLocn);

		// auto start = res.createRaw().data;
		int64_t offset = 0;
		res._offsetData(0);
		for (const auto &arr : arrays)
		{
			Array::applyUnaryOp(res, arr, ops::Copy(), true, offset);
			offset += step;
		}
		return res;
	}
}