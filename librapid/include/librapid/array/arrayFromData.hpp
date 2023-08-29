#ifndef LIBRAPID_ARRAY_FROM_DATA_HPP
#define LIBRAPID_ARRAY_FROM_DATA_HPP

namespace librapid {
	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE auto array::ArrayContainer<ShapeType, StorageType>::fromData(
	  const std::initializer_list<Scalar> &data) -> ArrayContainer {
		static_assert(!std::is_same_v<ShapeType, MatrixShape>,
					  "Cannot create a matrix from a 1D array");
		LIBRAPID_ASSERT(data.size() > 0, "Array must have at least one element");
		return ArrayContainer(data);
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE auto
	array::ArrayContainer<ShapeType, StorageType>::fromData(const std::vector<Scalar> &data)
	  -> ArrayContainer {
		static_assert(!std::is_same_v<ShapeType, MatrixShape>,
					  "Cannot create a matrix from a 1D array");
		LIBRAPID_ASSERT(data.size() > 0, "Array must have at least one element");
		return ArrayContainer(data);
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE auto array::ArrayContainer<ShapeType, StorageType>::fromData(
	  const std::initializer_list<std::initializer_list<Scalar>> &data) -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");

		if constexpr (std::is_same_v<ShapeType, MatrixShape>) {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
			auto res	  = ArrayContainer(newShape);
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT(data.begin()[i].size() == newShape[1],
								"Arrays must have consistent shapes");
				for (size_t j = 0; j < data.begin()[i].size(); ++j) {
					res(i, j) = data.begin()[i].begin()[j];
				}
			}
			return res;
		} else {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
#if defined(LIBRAPID_ENABLE_ASSERT)
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT(data.begin()[i].size() == newShape[1],
								"Arrays must have consistent shapes");
			}
#endif
			auto res	  = ArrayContainer(newShape);
			int64_t index = 0;
			for (const auto &item : data) res[index++] = fromData(item);
			return res;
		}
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE auto array::ArrayContainer<ShapeType, StorageType>::fromData(
	  const std::vector<std::vector<Scalar>> &data) -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");

		if constexpr (std::is_same_v<ShapeType, MatrixShape>) {
			auto newShape = ShapeType({data.size(), data[0].size()});
			auto res	  = ArrayContainer(newShape);
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT(data[i].size() == newShape[1],
								"Arrays must have consistent shapes");
				for (size_t j = 0; j < data[i].size(); ++j) { res(i, j) = data[i][j]; }
			}
			return res;
		} else {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
#if defined(LIBRAPID_ENABLE_ASSERT)
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT(data.begin()[i].size() == newShape[1],
								"Arrays must have consistent shapes");
			}
#endif
			auto res	  = ArrayContainer(newShape);
			int64_t index = 0;
			for (const auto &item : data) res[index++] = fromData(item);
			return res;
		}
	}

#define HIGHER_DIMENSIONAL_FROM_DATA(TYPE)                                                         \
	template<typename ShapeType, typename StorageType>                                             \
	LIBRAPID_ALWAYS_INLINE auto array::ArrayContainer<ShapeType, StorageType>::fromData(           \
	  const TYPE &data) -> ArrayContainer {                                                        \
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");                      \
		std::vector<ArrayContainer> tmp(data.size());                                              \
		int64_t index = 0;                                                                         \
		for (const auto &item : data) tmp[index++] = std::move(fromData(item));                    \
		auto zeroShape = tmp[0].shape();                                                           \
		for (int64_t i = 0; i < data.size(); ++i)                                                  \
			LIBRAPID_ASSERT(tmp[i].shape().operator==(zeroShape),                                  \
							"Arrays must have consistent shapes");                                 \
		auto newShape = ShapeType::zeros(zeroShape.ndim() + 1);                                    \
		newShape[0]	  = data.size();                                                               \
		for (size_t i = 0; i < zeroShape.ndim(); ++i) { newShape[i + 1] = zeroShape[i]; }          \
		auto res = Array<Scalar, Backend>(newShape);                                               \
		for (int64_t i = 0; i < data.size(); ++i) res[i] = tmp[i];                                 \
		return res;                                                                                \
	}

#define SINIT(SUB_TYPE) std::initializer_list<SUB_TYPE>
#define SVEC(SUB_TYPE)	std::vector<SUB_TYPE>

	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(Scalar))))
	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(SINIT(Scalar)))))
	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar)))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar))))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar)))))))))

	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(Scalar))))
	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(SVEC(Scalar)))))
	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar)))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar))))))))
	HIGHER_DIMENSIONAL_FROM_DATA(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar)))))))))

#undef SINIT
#undef HIGHER_DIMENSIONAL_FROM_DATA
} // namespace librapid

#endif // LIBRAPID_ARRAY_FROM_DATA_HPP