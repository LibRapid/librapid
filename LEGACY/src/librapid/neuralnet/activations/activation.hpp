#ifndef LIBRAPID_NEURALNET_ACTIVATIONS
#define LIBRAPID_NEURALNET_ACTIVATIONS

#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>

namespace librapid {
	/**
	 * \rst
	 *
	 * This type is inherited for all neural network activations, and implements
	 * the most basic features and functions.
	 *
	 * The activation type contains three functions: the activation, the
	 * derivative of the activation and the weight generator function
	 *
	 * .. Math::
	 *		\text{activation}=f(x)
	 *
	 *		\text{derivative}=\frac {df}{dx}f(x)
	 *
	 * .. Attention::
	 *		The inherited types are designed to be used as C++ pointers, as they
	 * 		can be "cast" to a ``basic_activation`` through polymorphism.
	 *		Passing in non-pointer activations to neural networks will result in
	 *		errors.
	 *
	 * \endrst
	 */
	class Activation {
	public:
		/**
		 * \rst
		 *
		 * Construct the activation to produce results that are correct
		 * for the layer it is being used on. This is important for generating
		 * the correct layer weight value distributions.
		 *
		 * Parameters
		 * ----------
		 *
		 * prevNodes: Integer
		 *		The number of nodes in the previous neural network layer
		 *
		 * Returns
		 * -------
		 *
		 * None
		 *
		 * \endrst
		 */
		virtual void construct(int64_t prevNodes) = 0;

		/**
		 * \rst
		 *
		 * The function ``f()`` is the activation function for the object.
		 * It takes in an array of values and returns a new array of the same
		 * size and datatype, where each value has been passed through the
		 *function :math:`f(x)`
		 *
		 * Parameters
		 * ----------
		 *
		 * arr: Array
		 *		The array to pass through the function :math:`f(x)`
		 *
		 * Returns
		 * -------
		 *
		 * result: Array
		 *		The array passed through :math:`f(x)`
		 *
		 * \endrst
		 */
		virtual Array f(const Array &arr) = 0;

		/**
		 * \rst
		 *
		 * The function ``df()`` is the derivative of the function ``f()``.
		 * It takes an array of values and returns a new array with the same
		 * dimensions and datatype, where every value has been passed through
		 * the function :math:`\frac{df}{dx}f(x)`.
		 *
		 * .. Attention::
		 *		To calculate the gradients correctly, the functions assume
		 *		the input array has already been "activated" (i.e. you are
		 *		calculating :math:`df(f(x))`)
		 *
		 * Parameters
		 * ----------
		 *
		 * arr: Array
		 *		The array whose values will be passed through the function
		 *		:math:`\frac{df}{dx}f(x)`
		 *
		 * Returns
		 * -------
		 *
		 * result: Array
		 *		The array passed through :math:`\frac{df}{dx}f(x)`
		 *
		 * \endrst
		 */
		virtual Array df(const Array &arr) = 0;

		/**
		 * \rst
		 *
		 * Create a new Array from a given size, where the values
		 * in the array are normalized to a specific range which is optimal for
		 * the activation function being used.
		 *
		 * .. Hint::
		 *		Using the weights generated from this function often leads
		 *		to faster, more efficient and more productive training of
		 *		neural network models
		 *
		 * Parameters
		 * ----------
		 *
		 * shape: extent
		 *		The shape of the weight to generate
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray
		 *		A new weight with normalized values in a particular range
		 *
		 * \endrst
		 */
		virtual Array weight(const Extent &e) const = 0;
	};
} // namespace librapid

#endif // LIBRAPID_NEURALNET_ACTIVATIONS