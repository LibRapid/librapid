#ifndef LIBRAPID_ARRAY_FROM_DATA_HPP
#define LIBRAPID_ARRAY_FROM_DATA_HPP

namespace librapid {
	/// \brief Create an array from a list of values (possibly multi-dimensional)
	///
	/// Create a new array from a potentially nested list of values. It is possible to specify the
	/// data type of the Array with the \p Scalar template parameter. If no type is specified, the
	/// type will be inferred from the data. The backend on which the Array is created can also be
	/// specified with the \p Backend template parameter. If no backend is specified, the Array will
	/// be created on the CPU.
	///
	/// \tparam Scalar The type of the Array
	/// \tparam Backend The backend on which the Array is created
	/// \param data The data from which the Array is created
	/// \return The created Array
	template<typename Scalar, typename Backend>
	auto array::ArrayContainer<Scalar, Backend>::fromData(const std::initializer_list<Scalar> &data)
	  -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Array must have at least one element");
		return ArrayContainer(data);
	}

	template<typename Scalar, typename Backend>
	auto array::ArrayContainer<Scalar, Backend>::fromData(const std::vector<Scalar> &data)
	  -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Array must have at least one element");
		return ArrayContainer(data);
	}

	template<typename Scalar, typename Backend>
	auto array::ArrayContainer<Scalar, Backend>::fromData(
	  const std::initializer_list<std::initializer_list<Scalar>> &data) -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");
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

	template<typename Scalar, typename Backend>
	auto
	array::ArrayContainer<Scalar, Backend>::fromData(const std::vector<std::vector<Scalar>> &data)
	  -> ArrayContainer {
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");
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

	//#define HIGHER_DIMENSIONAL_FROM_DATA(TYPE)                                                         \
//	template<typename Scalar, typename Backend>                                                    \
//	auto array::ArrayContainer<Scalar, Backend>::fromData(const TYPE &data) -> ArrayContainer {    \
//		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");                      \
//		auto *tmp	  = new ArrayContainer[data.size()];                                           \
//		int64_t index = 0;                                                                         \
//		for (const auto &item : data) tmp[index++] = fromData(item);                               \
//		auto zeroShape = tmp[0].shape();                                                           \
//		for (int64_t i = 0; i < data.size(); ++i)                                                  \
//			LIBRAPID_ASSERT(tmp[i].shape().operator==(zeroShape),                                  \
//							"Arrays must have consistent shapes");                                 \
//		auto newShape = ShapeType::zeros(zeroShape.ndim() + 1);                                    \
//		newShape[0]	  = data.size();                                                               \
//		for (size_t i = 0; i < zeroShape.ndim(); ++i) { newShape[i + 1] = zeroShape[i]; }          \
//		auto res = Array<Scalar, Backend>(newShape);                                               \
//		index	 = 0;                                                                              \
//		for (int64_t i = 0; i < data.size(); ++i) res[i] = std::move(tmp[i]);                      \
//		delete[] tmp;                                                                              \
//		return res;                                                                                \
//	}

#define HIGHER_DIMENSIONAL_FROM_DATA(TYPE)                                                         \
	template<typename Scalar, typename Backend>                                                    \
	auto array::ArrayContainer<Scalar, Backend>::fromData(const TYPE &data) -> ArrayContainer {    \
		LIBRAPID_ASSERT(data.size() > 0, "Cannot create a zero-sized array");                      \
		std::vector<ArrayContainer> tmp(data.size());                                              \
		int64_t index = 0;                                                                         \
		for (const auto &item : data) tmp[index++] = fromData(item);                               \
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