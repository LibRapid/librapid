#pragma once

namespace librapid {
	template<typename T, typename D>
	LR_INLINE Array<T, D> concatenate(const std::vector<Array<T, D>> &arrays, i64 axis = 0) {
		LR_ASSERT(arrays.size() > 0, "Cannot concatenate an empty array");

		// Quick return if possible
		if (arrays.size() == 1) { return arrays[0].copy(); }

		// Wrap the axis
		if (axis < 0) { axis += arrays[0].ndim(); }

		// Check all arrays have the same extent, other than the
		// concatenation axis
		i64 concatAxisSize = 0;
		Extent dim0		   = arrays[0].extent();
		for (i64 i = 1; i < arrays.size(); i++) {
			Extent dim = arrays[i].extent();
			LR_ASSERT(dim.ndim() == dim0.ndim(),
					  "All arrays must have the same number of dimensions");
			for (i64 j = 0; j < dim.ndim(); j++) {
				if (j == axis) { continue; }
				LR_ASSERT(dim[j] == dim0[j],
						  "All arrays must have the same shape, other than the concatenation axis");
			}
			concatAxisSize += dim[axis];
		}

		Extent resExtent(dim0.ndim() + 1);
		for (i64 i = 0; i < dim0.ndim(); i++) {
			if (i == axis) {
				resExtent[i]	 = concatAxisSize;
				resExtent[i + 1] = dim0[i];
			} else {
				resExtent[i + 1] = dim0[i];
			}
		}

		Array<T, D> result(resExtent);

		// Concatenate the arrays on the given axis
		for (i64 i = 0; i < arrays.size(); ++i) {}

		return 5;
	}

	// Same function, except for initializer lists
	template<typename T, typename D>
	LR_INLINE Array<T, D> concatenate(const std::initializer_list<Array<T, D>> &arrays,
									  i64 axis = 0) {
		return concatenate(std::vector(arrays), axis);
	}
} // namespace librapid