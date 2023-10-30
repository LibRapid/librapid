#ifndef LIBRAPID_ARRAY_FROM_DATA_HPP
#define LIBRAPID_ARRAY_FROM_DATA_HPP

namespace librapid::array {
	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType, StorageType>::ArrayContainer(
	  const std::initializer_list<Scalar> &data) :
			m_shape({data.size()}),
			m_size(data.size()), m_storage(StorageType::fromData(data)) {
		LIBRAPID_ASSERT_WITH_EXCEPTION(
		  std::invalid_argument, data.size() > 0, "Array must have at least one element");
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType, StorageType>::ArrayContainer::ArrayContainer(
	  const std::vector<Scalar> &data) :
			m_shape({data.size()}),
			m_size(data.size()), m_storage(StorageType::fromData(data)) {
		LIBRAPID_ASSERT_WITH_EXCEPTION(
		  std::invalid_argument, data.size() > 0, "Array must have at least one element");
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType, StorageType>::ArrayContainer(
	  const std::initializer_list<std::initializer_list<Scalar>> &data) {
		LIBRAPID_ASSERT_WITH_EXCEPTION(
		  std::invalid_argument, data.size() > 0, "Cannot create a zero-sized array");

		if constexpr (std::is_same_v<ShapeType, MatrixShape>) {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
			auto res	  = ArrayContainer(newShape);
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(
				  std::range_error,
				  data.begin()[i].size() == newShape[1],
				  "Arrays must have consistent shapes. {}th dimension had size {}, expected {}",
				  i,
				  data.begin()[i].size(),
				  newShape[1]);
				for (size_t j = 0; j < data.begin()[i].size(); ++j) {
					res(i, j) = data.begin()[i].begin()[j];
				}
			}

			// return res;
			*this = res;
		} else {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
#if defined(LIBRAPID_ENABLE_ASSERT)
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(
				  std::length_error,
				  data.begin()[i].size() == newShape[1],
				  "Arrays must have consistent shapes. {}th dimension had size {}, expected {}",
				  i,
				  data.begin()[i].size(),
				  newShape[1]);
			}
#endif
			auto res	  = ArrayContainer(newShape);
			int64_t index = 0;
			for (const auto &item : data) res[index++] = ArrayContainer(item);

			// return res;
			*this = res;
		}
	}

	template<typename ShapeType, typename StorageType>
	LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType, StorageType>::ArrayContainer(
	  const std::vector<std::vector<Scalar>> &data) {
		LIBRAPID_ASSERT_WITH_EXCEPTION(
		  std::invalid_argument, data.size() > 0, "Cannot create a zero-sized array");

		if constexpr (std::is_same_v<ShapeType, MatrixShape>) {
			auto newShape = ShapeType({data.size(), data[0].size()});
			auto res	  = ArrayContainer(newShape);
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(
				  std::range_error,
				  data[i].size() == newShape[1],
				  "Arrays must have consistent shapes. {}th dimension had size {}, expected {}",
				  i,
				  data[i].size(),
				  newShape[1]);
				for (size_t j = 0; j < data[i].size(); ++j) { res(i, j) = data[i][j]; }
			}

			// return res;
			*this = res;
		} else {
			auto newShape = ShapeType({data.size(), data.begin()->size()});
#if defined(LIBRAPID_ENABLE_ASSERT)
			for (size_t i = 0; i < data.size(); ++i) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(
				  std::range_error,
				  data.begin()[i].size() == newShape[1],
				  "Arrays must have consistent shapes. {}th dimension had size {}, expected {}",
				  i,
				  data.begin()[i].size(),
				  newShape[1]);
			}
#endif
			auto res	  = ArrayContainer(newShape);
			int64_t index = 0;
			for (const auto &item : data) res[index++] = ArrayContainer(item);

			// return res;
			*this = res;
		}
	}

#define HIGHER_DIMENSIONAL_FROM_DATA(TYPE)                                                         \
	template<typename ShapeType, typename StorageType>                                             \
	LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType, StorageType>::ArrayContainer(                 \
	  const TYPE &data) {                                                                          \
		LIBRAPID_ASSERT_WITH_EXCEPTION(                                                            \
		  std::invalid_argument, data.size() > 0, "Cannot create a zero-sized array");             \
		std::vector<ArrayContainer> tmp(data.size());                                              \
		int64_t index = 0;                                                                         \
		for (const auto &item : data) tmp[index++] = std::move(ArrayContainer(item));              \
		auto zeroShape = tmp[0].shape();                                                           \
		for (int64_t i = 0; i < data.size(); ++i)                                                  \
			LIBRAPID_ASSERT_WITH_EXCEPTION(                                                        \
			  std::range_error,                                                                    \
			  tmp[i].shape().operator==(zeroShape),                                                \
			  "Arrays must have consistent shapes. {}th dimension had {}. Expected {}",            \
			  i,                                                                                   \
			  tmp[i].shape(),                                                                      \
			  zeroShape);                                                                          \
		auto newShape = ShapeType::zeros(zeroShape.ndim() + 1);                                    \
		newShape[0]	  = data.size();                                                               \
		for (size_t i = 0; i < zeroShape.ndim(); ++i) { newShape[i + 1] = zeroShape[i]; }          \
		auto res = Array<Scalar, Backend>(newShape);                                               \
		for (int64_t i = 0; i < data.size(); ++i) res[i] = tmp[i];                                 \
		*this = res;                                                                               \
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
} // namespace librapid::array

#endif // LIBRAPID_ARRAY_FROM_DATA_HPP