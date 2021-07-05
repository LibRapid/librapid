#ifndef LIBRAPID_ACTIVATIONS
#define LIBRAPID_ACTIVATIONS

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>

namespace librapid
{
	namespace activations
	{
		/**
		 * \rst
		 *
		 * This type is inherited for all neural network activations, and
		 * implements the most basic features and functions.
		 *
		 * The activation type contains two functions, one of which is the
		 * activation function itself, and the other is the derivative.
		 *
		 * .. Math::
		 *		\text{activation}=f(x)
		 *
		 *		\text{derivative}=\frac {df}{dx}f(x)
		 *
		 * .. Attention::
		 *		The inherited types are designed to be used as C++ pointers,
		 *		as they can be "cast" to a ``basic_activation`` through
		 *		polymorphism. Passing in non-pointer activations to neural
		 *		networks will result in errors.
		 * \endrst
		 */
		template<typename T = double>
		class basic_activation
		{
		public:
			virtual ~basic_activation()
			{}

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
			 * prev_nodes: integer
			 *		The number of nodes in the previous neural network layer
			 *
			 * Returns
			 * -------
			 *
			 * None
			 *
			 * \endrst
			 */
			LR_INLINE virtual void construct(lr_int prev_nodes) = 0;

			/**
			 * \rst
			 *
			 * The function ``f()`` is the activation function for the object.
			 * It takes in an array of values and returns a new array of the same
			 * size and datatype, where each value has been passed through the function
			 * :math:`f(x)`
			 *
			 * Parameters
			 * ----------
			 *
			 * arr: basic_ndarray
			 *		The array to pass through the function :math:`f(x)`
			 *
			 * Returns
			 * -------
			 *
			 * result: basic_ndarray
			 *		The array passed through :math:`f(x)`
			 *
			 * \endrst
			 */
			LR_INLINE virtual basic_ndarray<T> f(const basic_ndarray<T> &arr) const = 0;

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
			 * arr: basic_ndarray
			 *		The array whose values will be passed through the function
			 *		:math:`\frac{df}{dx}f(x)`
			 *
			 * Returns
			 * -------
			 *
			 * result: basic_ndarray
			 *		The array passed through :math:`\frac{df}{dx}f(x)`
			 *
			 * \endrst
			 */
			LR_INLINE virtual basic_ndarray<T> df(const basic_ndarray<T> &arr) const = 0;

			/**
			 * \rst
			 *
			 * Create a new n-dimensional array from a given size, where the values
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
			LR_INLINE virtual basic_ndarray<T> weight(const extent &shape) const = 0;
		};

		/**
		 * \rst
		 *
		 * The sigmoid activation function
		 *
		 * .. Math::
		 *		\text{Activation: }
		 *		f(x)=\frac{1}{1 + e^{-x}}
		 *
		 *		\text{Derivative: }
		 *		\frac{df}{dx}f(x)=x\times(1-x)
		 *
		 *		\text{Weights: }
		 *		\text{Random distribution in range }\frac{\pm 1}{\sqrt{\text{previous_nodes}}}
		 *
		 * \endrst
		 */
		template<typename T = double>
		class sigmoid : public basic_activation<T>
		{
		public:
			LR_INLINE void construct(lr_int prev_nodes) override
			{
				m_prev_nodes = prev_nodes;
			}

			basic_ndarray<T> f(const basic_ndarray<T> &arr) const override
			{
				return (T) 1. / ((T) 1. + exp(-arr));
			}

			basic_ndarray<T> df(const basic_ndarray<T> &arr) const override
			{
				return arr * ((T) 1. - arr);
			}

			basic_ndarray<T> weight(const extent &shape) const override
			{
				T lower = -1. / std::sqrt((T) m_prev_nodes);
				T upper = 1. / std::sqrt((T) m_prev_nodes);
				basic_ndarray<T> res = basic_ndarray<T>(shape);
				res.fill_random(lower, upper);
				return res * (T) (upper - lower);
			}

		private:
			lr_int m_prev_nodes = 0;
		};

		/**
		 * \rst
		 *
		 * The sigmoid activation function
		 *
		 * .. Math::
		 *		\text{Activation: }
		 *		f(x)=tanh(x)
		 *
		 *		\text{Derivative: }
		 *		\frac{df}{dx}f(x)=1-x^2
		 *
		 *		\text{Weights: }
		 *		\text{Random distribution in range }\frac{\pm 1}{\sqrt{\text{previous_nodes}}}
		 *
		 * \endrst
		 */
		template<typename T = double>
		class tanh : public basic_activation<T>
		{
		public:
			LR_INLINE void construct(lr_int prev_nodes) override
			{
				m_prev_nodes = prev_nodes;
			}

			basic_ndarray<T> f(const basic_ndarray<T> &arr) const override
			{
				return librapid::tanh(arr);
			}

			basic_ndarray<T> df(const basic_ndarray<T> &arr) const override
			{
				return (T) 1. - (arr * arr);
			}

			basic_ndarray<T> weight(const extent &shape) const override
			{
				T lower = -1. / std::sqrt((T) m_prev_nodes);
				T upper = 1. / std::sqrt((T) m_prev_nodes);
				basic_ndarray<T> res = basic_ndarray<T>(shape);
				res.fill_random(lower, upper);
				return res * (T) (upper - lower);
			}

		private:
			lr_int m_prev_nodes = 0;
		};

		/**
		 * \rst
		 *
		 * The ReLU activation function
		 *
		 * .. Math::
		 *		\text{Activation: }
		 *		f(x)=
					\begin{cases}
						x\geq 0 &\quad x\\
						x < 0   &\quad 0.0\\
					\end{cases} \\

				\text{Derivative: }
				\frac{df}{dx}f(x)=
					\begin{cases}
						x\geq 0 &\quad 1.0\\
						x < 0   &\quad 0.0\\
					\end{cases}
		 *
		 *		\text{Weights: }
		 *		\text{random}[-1, 1] \times \sqrt{\frac{2}{\text{prev_nodes}}}
		 *
		 * \endrst
		 */
		template<typename T = double>
		class relu : public basic_activation<T>
		{
		public:
			LR_INLINE void construct(lr_int prev_nodes) override
			{
				m_prev_nodes = prev_nodes;
			}

			basic_ndarray<T> f(const basic_ndarray<T> &arr) const override
			{
				return maximum(arr, 0);
			}

			basic_ndarray<T> df(const basic_ndarray<T> &arr) const override
			{
				return greater_than(arr, 0);
			}

			basic_ndarray<T> weight(const extent &shape) const override
			{
				auto std = std::sqrt(2. / (T) m_prev_nodes);
				auto res = basic_ndarray<T>(shape);
				res.fill_random(-1, 1);
				return res * (T) std;
			}

		private:
			lr_int m_prev_nodes = 0;
		};

		/**
		 * \rst
		 *
		 * The ReLU activation function
		 *
		 * .. Math::
		 *		\text{Activation: }
		 *		f(x)=
					\begin{cases}
						x\geq 0 &\quad x\\
						x < 0   &\quad x \times 0.2\\
					\end{cases}

				\text{Derivative: }
				\frac{df}{dx}f(x)=
					\begin{cases}
						x\geq 0 &\quad 1.0\\
						x < 0   &\quad 0.2\\
					\end{cases}
		 *
		 *		\text{Weights: }
		 *		\text{random}[-1, 1] \times \sqrt{\frac{2}{\text{prev_nodes}}}
		 *
		 * \endrst
		 */
		template<typename T = double>
		class leaky_relu : public basic_activation<T>
		{
		public:
			LR_INLINE void construct(lr_int prev_nodes) override
			{
				m_prev_nodes = prev_nodes;
			}

			LR_INLINE basic_ndarray<T> f(const basic_ndarray<T> &arr) const override
			{
				return arr.mapped([](T x)
				{
					return x > 0 ? x : x * 0.2;
				});
			}

			LR_INLINE basic_ndarray<T> df(const basic_ndarray<T> &arr) const override
			{
				return arr.mapped([](T x)
				{
					return x > 0 ? 1 : 0.2;
				});
			}

			LR_INLINE basic_ndarray<T> weight(const extent &shape) const override
			{
				auto std = std::sqrt(2. / (T) m_prev_nodes);
				auto res = basic_ndarray<T>(shape);
				res.fill_random(-1, 1);
				return res * (T) std;
			}

		private:
			lr_int m_prev_nodes = 0;
		};
	}
}

#endif // LIBRAPID_ACTIVATIONS