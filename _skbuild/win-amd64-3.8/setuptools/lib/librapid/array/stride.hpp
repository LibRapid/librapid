#ifndef LIBRAPID_STRIDE
#define LIBRAPID_STRIDE

#include <librapid/config.hpp>
#include <librapid/array/extent.hpp>

namespace librapid
{
	class Stride
	{
	public:
		Stride() = default;

		/**
		 * \rst
		 *
		 * Create a Stride from the provided values. The number of dimensions will
		 * be equal to the number of elements in the provided list, and the stride
		 * for each dimension will be the corresponding value in the data provided
		 *
		 * \endrst
		 */
		template<typename _Ty = lr_int>
		Stride(const std::initializer_list<_Ty> &data)
			: Stride(std::vector<lr_int>(data.begin(), data.end()))
		{}

		template<typename _Ty = lr_int,
			typename std::enable_if<std::is_integral<_Ty>::value, int>::type = 0>
			Stride(const std::vector<_Ty> &data)
		{
			// Initialize members
			m_isTrivial = true;
			m_isContiguous = true;
			m_dims = data.size();

			// Check for a valid number of dimensions
			if (m_dims > LIBRAPID_MAX_DIMS)
				throw std::runtime_error("Cannot create Stride with "
										 + std::to_string(m_dims)
										 + " dimensions. Maximum allowed is "
										 + std::to_string(LIBRAPID_MAX_DIMS));

			for (size_t i = 0; i < data.size(); ++i)
				m_stride[i] = data[i];
		}

		/**
		 * \rst
		 *
		 * Create a Stride from another Stride. This will copy all data, including
		 * trivial-ness and contiguity.
		 *
		 * \endrst
		 */
		Stride(const Stride &other)
		{
			m_isTrivial = other.m_isTrivial;
			m_isContiguous = other.m_isContiguous;
			m_dims = other.m_dims;

			for (size_t i = 0; i < m_dims; ++i)
				m_stride[i] = other.m_stride[i];
		}

	#ifdef LIBRAPID_PYTHON
		Stride(py::args args)
		{
			m_dims = py::len(args);

			if (m_dims > LIBRAPID_MAX_DIMS)
				throw std::runtime_error("Cannot create Stride with "
										 + std::to_string(m_dims)
										 + " dimensions. Limit is "
										 + std::to_string(LIBRAPID_MAX_DIMS));

			size_t neg = 0;

			for (lr_int i = 0; i < m_dims; i++)
				m_stride[i] = py::cast<lr_int>(args[i]);
		}
	#endif

		/**
		 * \rst
		 *
		 * Set one Stride equal to another. This will copy all data, including
		 * trivial-ness and contiguity.
		 *
		 * \endrst
		 */
		LR_INLINE Stride &operator=(const Stride &other)
		{
			m_dims = other.m_dims;
			m_isTrivial = other.m_isTrivial;
			m_isContiguous = other.m_isContiguous;

			for (size_t i = 0; i < m_dims; ++i)
			{
				m_stride[i] = other.m_stride[i];
			}

			return *this;
		}

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
		LR_INLINE static Stride fromExtent(const Extent &extent)
		{
			Stride res;
			res.m_dims = extent.ndim();

			size_t prod = 1;
			for (size_t i = 0; i < extent.ndim(); ++i)
			{
				res.m_stride[res.m_dims - i - 1] = (lr_int) prod;
				prod *= extent[res.m_dims - i - 1];
			}

			return res;
		}

		/**
		 * \rst
		 *
		 * Return the number of dimension of the Stride
		 *
		 * \endrst
		 */
		LR_INLINE size_t ndim() const
		{
			return m_dims;
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
		LR_INLINE bool isTrivial() const
		{
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
		LR_INLINE bool isContiguous() const
		{
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
		LR_INLINE bool operator==(const Stride &other) const
		{
			if (m_dims != other.m_dims)
				return false;

			if (m_isTrivial != other.m_isTrivial)
				return false;

			if (m_isContiguous != other.m_isContiguous)
				return false;

			for (size_t i = 0; i < m_dims; ++i)
				if (m_stride[i] != other.m_stride[i])
					return false;

			return true;
		}

		/**
		 * \rst
		 *
		 * Returns true if two Strides are *not* equal.
		 *
		 * See ``Stride::operator==(Stride)``
		 *
		 * \endrst
		 */
		LR_INLINE bool operator!=(const Stride &other) const
		{
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
		LR_INLINE const lr_int &operator[](const size_t index) const
		{
			if (index > m_dims)
				throw std::out_of_range("Cannot access index "
										+ std::to_string(index)
										+ " of Stride with "
										+ std::to_string(m_dims) + " dimensions");

			return m_stride[index];
		}

		LR_INLINE lr_int &operator[](const size_t index)
		{
			if (index > m_dims)
				throw std::out_of_range("Cannot access index "
										+ std::to_string(index)
										+ " of Stride with "
										+ std::to_string(m_dims) + " dimensions");

			return m_stride[index];
		}

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
		LR_INLINE void reorder(const std::vector<size_t> &order)
		{
			Stride temp = *this;

			for (size_t i = 0; i < order.size(); ++i)
				m_stride[i] = temp.m_stride[order[i]];

			m_isTrivial = checkTrivial();
		}

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
		LR_INLINE bool checkTrivial() const
		{
			// Ensure every stride is bigger than the next one
			bool foundOne = false;
			for (size_t i = 0; i < m_dims; ++i)
			{
				if (m_stride[i] <= m_stride[i + 1]) return false;
				if (m_stride[i] == 1) foundOne = true;
			}

			return foundOne;
		}

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
		LR_INLINE bool checkContiguous(const Extent &extent) const
		{
			if (m_dims != extent.ndim())
				throw std::domain_error("Stride and Extent must have the same "
										"dimensions for a contiguity test");

			Stride temp = fromExtent(extent);
			size_t valid = 0;

			for (size_t i = 0; i < m_dims; ++i)
			{
				for (size_t j = 0; j < m_dims; j++)
				{
					if (temp[i] == m_stride[i])
					{
						++valid;
						break;
					}
				}
			}

			return valid == m_dims;
		}

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
		LR_INLINE std::string str() const
		{
			std::stringstream res;
			res << "Stride(";
			for (size_t i = 0; i < m_dims; ++i)
			{
				res << m_stride[i];

				if (i < m_dims - 1)
					res << ", ";
			}
			res << ")";

			return res.str();
		}

		LR_INLINE ESIterator begin() const
		{
			return ESIterator((lr_int *) m_stride);
		}

		LR_INLINE ESIterator end() const
		{
			return ESIterator((lr_int *) m_stride + m_dims);
		}

	private:

		lr_int m_stride[LIBRAPID_MAX_DIMS]{};
		size_t m_dims = 0;
		bool m_isTrivial = true; // Trivial stride
		bool m_isContiguous = true; // Data is contiguous in memory
	};
}

#endif // LIBRAPID_STRIDE