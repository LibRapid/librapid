#ifndef LIBRAPID_STRIDE
#define LIBRAPID_STRIDE

#include <librapid/config.hpp>
#include <librapid/array/extent.hpp>

namespace librapid {
	class Stride {
	public:
		Stride() = default;

		/**
		 * \rst
		 *
		 * Create a Stride from the provided values. The number of dimensions will
		 * be equal to the number of elements in the provided list, and the stride
		 * for each dimension will be the corresponding value in the data provided
		 *
		 * If the provided value is a single scalar, the Stride object will
		 * be created with that number of dimensions, with each value set to one.
		 *
		 * .. Attention::
		 *
		 *		Creation from a scalar value only works in C++, as in Python the
		 *		arguments can be passed without specifying they are a list
		 *
		 *		Example:
		 *
		 *		>>> librapid.Stride([5])
		 *		<librapid.Stride(5)>
		 *
		 *		>>> librapid.Stride(5) # Note, not a list
		 *		<librapid.Stride(5)>
		 *
		 * \endrst
		 */
		explicit Stride(const std::initializer_list<int64_t> &data);

		explicit Stride(const std::vector<int64_t> &data);

		explicit Stride(int64_t dims);

#ifdef LIBRAPID_PYTHON
		Stride(py::args args);
#endif

		/**
		 * \rst
		 *
		 * Create a new Stride from the provided Extent object. The number of
		 * dimensions of the generated Stride will match those of the Extent
		 * provided, and each value for the Stride will be calculated to fit the
		 * dimensions of the Extent.
		 *
		 * The algorithm for generating the stride is as follows:
		 *
		 * .. code-block:: python
		 *		:caption: Python code for generating a Stride from an Extent
		 *
		 *		Stride res
		 * 		res.dims = extent.ndim()
		 *
		 *		for i in range(extent.ndim()):
		 *			res[res.ndim() - i - 1] = prod
		 * 			prod *= extent[ndim() - i - 1]
		 *
		 * \endrst
		 */
		static Stride fromExtent(const Extent &extent);

		void setTrivial(bool newVal);

		void setContiguity(bool newVal);

		/**
		 * \rst
		 *
		 * Return the number of dimension of the Stride
		 *
		 * \endrst
		 */
		inline int64_t ndim() const {
			return m_dims;
		}

		/**
		* \rst
		*
		* Return a pointer to the raw data of this stride
		*
		* \endrst
		*/
		inline const int64_t *__restrict raw() const {
			return m_stride;
		}

		/**
		 * \rst
		 *
		 * Convert the Stride object to an std::vector and return the result
		 *
		 * \endrst
		 */
		inline std::vector<int64_t> toVec() const {
			std::vector<int64_t> res(m_dims);
			for (int64_t i = 0; i < m_dims; ++i)
				res[i] = m_stride[i];
			return res;
		}

		/**
		 * \rst
		 *
		 * Return whether or not the Stride is trivial or not.
		 *
		 * .. Hint::
		 *
		 *		See ``Stride.checkTrivial()``
		 *
		 * \endrst
		 */
		inline bool isTrivial() const {
			return m_isTrivial;
		}

		/**
		 * \rst
		 *
		 * Return whether the Stride represents a contiguous memory block
		 *
		 * .. Hint::
		 *
		 *		See ``Stride.checkContiguous(Extent)``
		 *
		 * \endrst
		 */
		inline bool isContiguous() const {
			return m_isContiguous;
		}

		/**
		 * \rst
		 *
		 * Returns true if the Stride matches another Stride *exactly*. This
		 * requires the dimensions, trivial-ness and contiguity to match, as well as
		 * the actual values for the stride.
		 *
		 * \endrst
		 */
		bool operator==(const Stride &other) const;

		/**
		 * \rst
		 *
		 * Returns true if two Strides are *not* equal.
		 *
		 * See ``Stride::operator==(Stride)``
		 *
		 * \endrst
		 */
		inline bool operator!=(const Stride &other) const {
			return !(*this == other);
		}

		/**
		 * \rst
		 *
		 * Access the value at a given index in the Stride.
		 *
		 * An error will be thrown if the index is out of bounds for the Stride
		 *
		 * \endrst
		 */
		const int64_t &operator[](const int64_t index) const;

		int64_t &operator[](const int64_t index);

		/**
		 * \rst
		 *
		 * Reorder the contents of the Stride object. This has the effect of
		 * transposing the Array it represents.
		 *
		 * .. Attention::
		 *
		 *		Each index must appear only once. There is currently no check on
		 *		this, so it should be handled by the calling function.
		 *
		 * Parameters
		 * ----------
		 *
		 * order: list
		 *		A list of indices representing the order in which to reshape
		 *
		 * \endrst
		 */
		void reorder(const std::vector<int64_t> &order);

		/**
		 * \rst
		 *
		 * Return a new Stride object containing the values from this Stride in the
		 * range :math:`[\text{start}, \text{end})`
		 *
		 * \endrst
		 */
		Stride subStride(int64_t start = -1, int64_t end = -1) const;

		// void scaleBytes(int64_t bytes);
		// Stride scaledBytes(int64_t bytes) const;

		/**
		 * \rst
		 *
		 * Returns true if the Stride is roughly trivial
		 *
		 * .. Hint::
		 *
		 *		A stride is trivial if every value in the stride is greater than
		 *		the following value. This is a crude method of detecting what
		 *		algorithms can be used on a given Array, but, when combined with the
		 *		``isContiguous()`` function, provides an effective classifier for
		 *		algorithm selection.
		 *
		 * \endrst
		 */
		bool checkTrivial() const;

		/**
		 * \rst
		 *
		 * Returns true if the Stride and Extent (passed as a parameter) represent
		 * an Array whose data is contiguous in memory.
		 *
		 * Let :math:`A` be an array with extent :math:`E=\{E_n, E_{n-1}, ... E_2, E_1\}`
		 * and stride :math:`S=\{S_n, S_{n-1}, ... S_2, S_1\}`. :math:`A` is
		 * contiguous in memory if, and only if, :math:`S' \cap D' = \emptyset`,
		 * where
		 *
		 * .. Math::
				D_n = \begin{cases}
					x \geq 2  &\quad \prod_{i=2}^{\text{dims}_a}{S_i} \\
					otherwise &\quad 1 \\
				\end{cases}
		 *
		 * \endrst
		 */
		bool checkContiguous(const Extent &extent) const;

		/**
		* \rst
		*
		* Generate a string representation of the Stride. The result takes the
		* following general form:
		*
		* :math:`\text{Stride}(\text{stride}_0, \text{stride}_1, ... , \text{stride}_{n-1})`
		*
		* \endrst
		*/
		std::string str() const;

		inline ESIterator begin() const {
			return ESIterator((int64_t *) m_stride);
		}

		inline ESIterator end() const {
			return ESIterator((int64_t *) m_stride + m_dims);
		}

	private:

		int64_t m_stride[LIBRAPID_MAX_DIMS];
		int64_t m_dims = 0;
		bool m_isTrivial = true; // Trivial stride
		bool m_isContiguous = true; // Data is contiguous in memory
	};

	inline std::ostream &operator<<(std::ostream &os, const Stride &stride) {
		return os << stride.str();
	}
}

#endif // LIBRAPID_STRIDE