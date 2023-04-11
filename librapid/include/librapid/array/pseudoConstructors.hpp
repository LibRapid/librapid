#ifndef LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP
#define LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP

namespace librapid {
	/// \brief Create an Array filled with zeros
	///
	/// Create an array with a specified shape, scalar type and device, and fill it with zeros.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Device Device type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with zeros
	template<typename Scalar = double, typename Device = device::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Device> zeros(const Shape<T, N> &shape) {
		return Array<Scalar, Device>(shape, Scalar(0));
	}

	/// \brief Create an Array filled with ones
	///
	/// Create an array with a specified shape, scalar type and device, and fill it with ones.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Device Device type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with ones
	template<typename Scalar = double, typename Device = device::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Device> ones(const Shape<T, N> &shape) {
		return Array<Scalar, Device>(shape, Scalar(1));
	}

	/// \brief Create an Array filled, in order, with the numbers 0 to N-1
	///
	/// Create a new Array object with a given shape, where each value is filled with a number from
	/// 0 to N-1, where N is the total number of elements in the array. The values are filled in
	/// the same order as the array is stored in memory.
	///
	/// \tparam Scalar Scalar type of the Array
	/// \tparam Device Device type of the Array
	/// \tparam T Scalar type of the Shape
	/// \tparam N Maximum number of dimensions of the Shape
	/// \param shape Shape of the Array
	/// \return Array filled with numbers from 0 to N-1
	template<typename Scalar = int64_t, typename Device = device::CPU, typename T = size_t, size_t N = 32>
	Array<Scalar, Device> ordered(const Shape<T, N> &shape) {
		Array<Scalar, Device> result(shape);
		for (size_t i = 0; i < shape.size(); i++) { result.storage()[i] = Scalar(i); }
		return result;
	}

	/// \brief Create a 1-dimensional Array from a range of numbers and a step size
	///
	/// Provided with a start value and a stop value,
	/// \tparam T
	/// \tparam Scalar
	/// \tparam Device
	/// \param start
	/// \param stop
	/// \param step
	/// \return
	template<typename T, typename Scalar = double, typename Device = device::CPU>
	Array<Scalar, Device> arange(T start, T stop, T step) {
		LIBRAPID_ASSERT(step != 0, "Step size cannot be zero");
		LIBRAPID_ASSERT((stop - start) / step > 0, "Step size is invalid for the specified range");

		Shape shape = {(int64_t)((stop - start) / step)};
		Array<Scalar, Device> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = Scalar(start + i * step);
		}
		return result;
	}

	template<typename Scalar = double, typename Device = device::CPU, typename T>
	Array<Scalar, Device> arange(T start, T stop) {
		LIBRAPID_ASSERT((stop - start) > 0, "Step size is invalid for the specified range");

		Shape shape = {(int64_t)((stop - start))};
		Array<Scalar, Device> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = Scalar(start + i);
		}
		return result;
	}

	template<typename Scalar = double, typename Device = device::CPU, typename T>
	Array<Scalar, Device> arange(T stop) {
		Shape shape = {(int64_t)(stop)};
		Array<Scalar, Device> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = Scalar(i);
		}
		return result;
	}

	template<typename Scalar = double, typename Device = device::CPU, typename T>
	Array<Scalar, Device> linspace(T start, T stop, int64_t num) {
		LIBRAPID_ASSERT(num > 0, "Number of samples must be greater than zero");

		Shape shape = {num};
		Array<Scalar, Device> result(shape);
		for (size_t i = 0; i < shape.size(); i++) {
			result.storage()[i] = Scalar(start + (stop - start) * i / (num - 1));
		}
		return result;
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_PSEUDO_CONSTRUCTORS_HPP