#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid {
	void Array::transpose(const Extent &order) {
		Extent tempOrder;

		if (order.ndim() == 0) {
			// Order was not provided, so initialize it to be n, n-1, ... , 2, 0
			tempOrder = Extent(m_extent.ndim());
			for (int64_t i = 0; i < m_extent.ndim(); ++i)
				tempOrder[i] = m_extent.ndim() - i - 1;
		} else {
			tempOrder = order;
		}

		// Check the dimensions are correct
		if (tempOrder.ndim() != m_extent.ndim()) {
			throw std::invalid_argument(std::to_string(tempOrder.ndim()) + " indices "
																		   "were passed to Array transpose, though "
										+ std::to_string(m_extent.ndim()) +
										" indices are required");
		}

		bool valid = true;
		int64_t count = 0;
		int64_t missing[LIBRAPID_MAX_DIMS]{};
		for (int64_t i = 0; i < ndim(); ++i) {
			if (std::count(&(*tempOrder.begin()), &(*tempOrder.end()), i) != 1) {
				missing[count++] = i;
				valid = false;
			}
		}

		if (!valid) {
			auto stream = std::stringstream();
			for (int64_t i = 0; i < count; ++i) {
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

	void Array::reshape(const Extent &newShape) {
		if (m_isChild)
			throw std::runtime_error("Cannot reshape child array. Either "
									 "assign it to a variable or copy the value first");

		if (!m_isScalar && (newShape.ndim() == 0 || (newShape.ndim() == 1 && newShape[0] == 0))) {
			m_isScalar = true;
			m_extent = Extent({1});
			m_stride = Stride({1});
		}

		auto tempShape = newShape.fixed(m_extent.size());

		if (tempShape.size() != m_extent.size())
			throw std::domain_error("Cannot reshape array with "
									+ m_extent.str() + " (" + std::to_string(m_extent.size())
									+ " elements) to array with " + tempShape.str()
									+ " (" + std::to_string(m_extent.size()) + " elements");

		if (!m_stride.isTrivial() || !m_stride.isContiguous()) {
			Array res(tempShape, m_dtype, m_location);
			Array::applyUnaryOp(res, *this, ops::Copy(), true);
			m_extent = res.m_extent;
			m_stride = res.m_stride;
			decrement();
			m_dataOrigin = res.m_dataOrigin;
			m_dataStart = res.m_dataStart;
			res.increment();
			return;
		}

		m_extent = tempShape;
		m_stride = Stride::fromExtent(tempShape);
	}

	Array concatenate(const std::vector<Array> &arrays, int64_t axis) {
		if (arrays.empty()) return Array();
		if (arrays.size() == 1) return arrays[0].clone();

		// Check all arrays are the same size and compute the new dimension for the
		// resulting array
		int64_t index = 0;
		int64_t dims = 0;
		Datatype resDtype = arrays[0].dtype();
		Accelerator resLocn = arrays[0].location();

		for (uint64_t i = 0; i < arrays.size(); ++i) {
			if (arrays[i].ndim() > dims) {
				dims = arrays[i].ndim();
				index = i;
			}
			if (arrays[i].dtype() > resDtype) resDtype = arrays[i].dtype();
			if (arrays[i].location() > resLocn) resLocn = arrays[i].location();
		}

		if (axis < 0 || axis > dims)
			throw std::range_error("Axis " + std::to_string(axis)
								   + " is out of range for array with "
								   + std::to_string(dims) + " dimensions");

		int64_t newDim = arrays[index].extent()[axis];
		const Extent &dim0 = arrays[index].extent();

		for (uint64_t i = 1; i < arrays.size(); ++i) {
			bool adjustCheck = false;
			if (arrays[i].ndim() != dims && !(adjustCheck = arrays[i].ndim() == dims - 1))
				throw std::invalid_argument("To concatenate arrays, all "
											"values must have the same number of "
											"dimensions, however (at least) one "
											"array failed this condition. "
											+ std::to_string(arrays[i].ndim())
											+ " dimension(s) is not compatible with "
											+ std::to_string(dims) + " dimension(s)");

			// Ensure every dimension other than <axis> is equal
			// (concatenating on <axis> allows for different size in that dimension)
			for (int64_t j = 0; j < dims - adjustCheck; ++j) {
				if (j != axis && arrays[i].extent()[j] != dim0[j + (adjustCheck * (j > axis))])
					throw std::invalid_argument("To concatenate arrays, all "
												"dimensions other than index <axis> "
												"must be equal, however (at least) "
												"one array failed this condition. "
												+ arrays[i].extent().str()
												+ " is not compatible with "
												+ dim0.str());
			}

			newDim += adjustCheck ? 1 : arrays[i].extent()[axis];
		}

		int64_t step = 1;
		for (int64_t d = axis; d < dims; ++d)
			step *= dim0[d];

		Extent resShape = dim0;
		resShape[axis] = newDim;
		// resShape.update();

		Array res(resShape, resDtype, resLocn);

		int64_t offset = 0;
		if (axis) res._offsetData(0);

		Extent fixed = dim0;
		fixed[axis] = -1;
		for (const auto &arr: arrays) {
			if (arr.ndim() == dims)
				Array::applyUnaryOp(res, arr, ops::Copy(), true, offset);
			else if (arr.ndim() == dims - 1)
				Array::applyUnaryOp(res, arr.reshaped(fixed), ops::Copy(), true, offset);
			offset += step;
		}
		res._setScalar(false);

		return res;
	}

	Array stack(const std::vector<Array> &arrays, int64_t axis) {
		int64_t newDim = arrays.size();
		int64_t dims = arrays[0].ndim();

		// Perform some checks to make sure everything is going to work
		if (axis < 0 || axis > arrays[0].ndim() + 1)
			throw std::invalid_argument("Axis " + std::to_string(axis)
										+ " is out of range for arrays with "
										+ std::to_string(arrays[0].ndim()) + " dimensions");

		if (newDim == 0) return Array();
		if (newDim == 1) return arrays[0].clone();

		Extent dim0 = arrays[0].extent();
		for (int64_t i = 0; i < newDim; ++i) {
			if (arrays[i].extent() != dim0)
				throw std::invalid_argument("Array with " + arrays[i].extent().str()
											+ " cannot be stacked with array with " + dim0.str()
											+ ". All arrays must have the same extent");
		}

		Extent resShape(arrays[0].isScalar() ? 1 : dims + 1);
		if (arrays[0].isScalar()) {
			resShape[0] = arrays.size();
		} else {
			for (int64_t i = 0; i < dims + 1; ++i) {
				if (i < axis) resShape[i] = dim0[i];
				else if (i == axis) resShape[i] = newDim;
				else resShape[i] = dim0[i - 1];
			}
		}
		resShape.update();

		// If we're adding to the last dimension, reshape the arrays to ensure
		// that everything will turn out correctly
		if (axis == dims) {
			Extent tempShape(dims + 1);
			for (int64_t i = 0; i < dims + 1; ++i) {
				if (i == axis) tempShape[i] = 1;
				else tempShape[i] = dim0[i];
			}
			tempShape.update();

			std::vector<Array> tempArrays(arrays.size());
			for (uint64_t i = 0; i < arrays.size(); ++i)
				tempArrays[i] = arrays[i].reshaped(tempShape);

			return concatenate(tempArrays, axis).reshaped(resShape);
		}

		return concatenate(arrays, axis).reshaped(resShape);
	}
}