#ifndef LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP
#define LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP

namespace librapid {
	/// \brief Create an Array filled with zeros
	///
	/// Create an array with a specified shape, scalar type and Backend, and fill it with zeros.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Backend Backend type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with zeros
	template<typename Scalar = double, typename Backend = backend::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Backend> zeros(const Shape<T, N> &shape) {
		return Array<Scalar, Backend>(shape, Scalar(0));
	}

	/// \brief Create an Array filled with ones
	///
	/// Create an array with a specified shape, scalar type and Backend, and fill it with ones.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Backend Backend type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with ones
	template<typename Scalar = double, typename Backend = backend::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Backend> ones(const Shape<T, N> &shape) {
		return Array<Scalar, Backend>(shape, Scalar(1));
	}

	/// \brief Create an Array filled, in order, with the numbers 0 to N-1
	///
	/// Create a new Array object with a given shape, where each value is filled with a number from
	/// 0 to N-1, where N is the total number of elements in the array. The values are filled in
	/// the same order as the array is stored in memory.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Backend Backend type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with numbers from 0 to N-1
	template<typename Scalar = int64_t, typename Backend = backend::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Backend> ordered(const Shape<T, N> &shape) {
		Array<Scalar, Backend> result(shape);
		for (size_t i = 0; i < shape.size(); i++) { result.storage()[i] = Scalar(i); }
		return result;
	}

	/// \brief Create a 1-dimensional Array from a range of numbers and a step size
	///
	/// Provided with a start value and a stop value, create a 1-dimensional Array with
	/// \f$\lfloor \frac{stop - start}{step} \rfloor \f$ elements, where each element is
	/// \f$start + i \times step\f$, for \f$i \in [0, \lfloor \frac{stop - start}{step} \rfloor)\f$.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Backend Backend for the Array
	/// \tparam T Scalar type of the start, stop and step values
	/// \param start First value in the range
	/// \param stop Second value in the range
	/// \param step Step size between values in the range
	/// \return Array
	template<typename Scalar = double, typename Backend = backend::CPU, typename T>
	Array<Scalar, Backend> arange(T start, T stop, T step) {
		LIBRAPID_ASSERT(step != 0, "Step size cannot be zero");
		LIBRAPID_ASSERT((stop - start) / step > 0, "Step size is invalid for the specified range");

		Shape shape = {(int64_t)::librapid::abs((stop - start) / step)};
		Array<Scalar, Backend> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = Scalar(start + i * step);
		}
		return result;
	}

	/// \brief Create a 1-dimensional Array from a range of numbers
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Backend Backend for the Array
	/// \tparam T Scalar type of the start and stop values
	/// \param start Lower bound of the range
	/// \param stop Upper bound of the range
	/// \return Array
	/// \see arange(T start, T stop, T step)
	template<typename Scalar = double, typename Backend = backend::CPU, typename T>
	Array<Scalar, Backend> arange(T start, T stop) {
		LIBRAPID_ASSERT((stop - start) > 0, "Step size is invalid for the specified range");

		Shape shape = {(int64_t)::librapid::abs(stop - start)};
		Array<Scalar, Backend> result(shape);
		for (size_t i = 0; i < shape.size(); i++) { result.storage()[i] = Scalar(start + i); }
		return result;
	}

	/// \brief Create a 1-dimensional Array
	/// \tparam Scalar
	/// \tparam Backend
	/// \tparam T
	/// \param stop
	/// \return
	template<typename Scalar = double, typename Backend = backend::CPU, typename T>
	Array<Scalar, Backend> arange(T stop) {
		Shape shape = {(int64_t)::librapid::abs(stop)};
		Array<Scalar, Backend> result(shape);
		for (size_t i = 0; i < shape.size(); i++) { result.storage()[i] = Scalar(i); }
		return result;
	}

	template<typename Scalar = double, typename Backend = backend::CPU, typename Start, typename Stop>
	Array<Scalar, Backend> linspace(Start start, Stop stop, int64_t num, bool includeEnd = true) {
		LIBRAPID_ASSERT(num > 0, "Number of samples must be greater than zero");

		auto startCast = static_cast<Scalar>(start);
		auto stopCast  = static_cast<Scalar>(stop);
		auto den	   = static_cast<Scalar>(num - includeEnd);
		Shape shape	   = {num};
		Array<Scalar, Backend> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = startCast + (stopCast - startCast) * static_cast<Scalar>(i) / den;
		}
		return result;
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP