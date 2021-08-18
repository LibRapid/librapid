#ifndef LIBRAPID_EXTENT
#define LIBRAPID_EXTENT

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/iterators.hpp>

namespace librapid
{
	class Extent
	{
	public:
		/**
		 * \rst
		 *
		 * Default Extent constructor. The Extent has 0 dimensions by default.
		 *
		 * \endrst
		 */
		Extent() = default;

		/**
		 * \rst
		 *
		 * Construct an Extent from the provided data, where the number of elements
		 * passed defines the *number* of dimensions of the Extent, and the integer
		 * values passed represent the size of each dimension respectively.
		 *
		 * Parameters
		 * ----------
		 *
		 * data: list type
		 *		The Extents data
		 *
		 * \endrst
		 */
		template<typename _Ty = lr_int>
		Extent(const std::initializer_list<_Ty> &data)
			: Extent(std::vector<lr_int>(data.begin(), data.end()))
		{}

		template<typename _Ty = lr_int,
			typename std::enable_if<std::is_integral<_Ty>::value, int>::type = 0>
			Extent(const std::vector<_Ty> &data)
		{
			// Initialize the dimensions
			m_dims = data.size();

			if (m_dims > LIBRAPID_MAX_DIMS)
				throw std::runtime_error("Cannot create Extent with "
										 + std::to_string(m_dims)
										 + " dimensions. Limit is "
										 + std::to_string(LIBRAPID_MAX_DIMS));

			for (size_t i = 0; i < data.size(); i++)
				m_extent[i] = data[i];

			update();
		}

		/**
		 * \rst
		 *
		 * Create an Extent from a given number of dimensions. The size for each
		 * dimension will be set to 1 by default.
		 *
		 * Parameters
		 * ----------
		 *
		 * dims : integer
		 *		Number of dimensions for Extent
		 *
		 * \endrst
		 */
		Extent(size_t dims)
		{
			m_dims = dims;
			m_size = dims;
			if (m_dims > LIBRAPID_MAX_DIMS)
				throw std::runtime_error("Cannot create Extent with "
										 + std::to_string(m_dims)
										 + " dimensions. Limit is "
										 + std::to_string(LIBRAPID_MAX_DIMS));

			for (size_t i = 0; i < m_dims; ++i)
				m_extent[i] = 1;
		}

		Extent(const Extent &other)
		{
			memcpy(m_extent, other.m_extent, sizeof(lr_int) * LIBRAPID_MAX_DIMS);
			m_dims = other.m_dims;
			m_size = other.m_size;
			m_containsAutomatic = other.m_containsAutomatic;

			update();
		}

	#ifdef LIBRAPID_PYTHON
		Extent(py::args args)
		{
			m_dims = py::len(args);

			if (m_dims > LIBRAPID_MAX_DIMS)
				throw std::runtime_error("Cannot create Extent with "
										 + std::to_string(m_dims)
										 + " dimensions. Limit is "
										 + std::to_string(LIBRAPID_MAX_DIMS));

			for (lr_int i = 0; i < m_dims; i++)
				m_extent[i] = py::cast<lr_int>(args[i]);

			update();
		}
	#endif

		LR_INLINE Extent &operator=(const Extent &other)
		{
			memcpy(m_extent, other.m_extent, sizeof(lr_int) * LIBRAPID_MAX_DIMS);
			m_dims = other.m_dims;
			m_size = other.m_size;
			m_containsAutomatic = other.m_containsAutomatic;

			update();

			return *this;
		}

		/**
		 * \rst
		 *
		 * Get the number of dimensions of the Extent
		 *
		 * \endrst
		 */
		LR_INLINE const size_t &ndim() const
		{
			return m_dims;
		}

		/**
		 * \rst
		 *
		 * Returns the number of elements the Extent object represents. This is the
		 * product ``dim1 * dim2 * dim3 ... ``.
		 *
		 * .. Attention::
		 *
		 *		If an automatic dimension (``librapid.AUTO``) is included in the
		 *		Extent, the size value will be negative. Ensure your program takes
		 *		this into account
		 *
		 * \endrst
		 */
		LR_INLINE lr_int size() const
		{
			return m_size;
		}

		/**
		 * \rst
		 *
		 * Returns true if the Extent contains an automatic dimension.
		 *
		 * .. Hint::
		 *
		 *		An automatic dimension is any negative value or ``librapid.AUTO``
		 *
		 * \endrst
		 */
		LR_INLINE bool containsAutomatic() const
		{
			return m_containsAutomatic;
		}

		/**
		 * \rst
		 *
		 * Return the size of the Extent at a given index.
		 *
		 * If the index is out of range, an error will be thrown.
		 *
		 * Parameters
		 * ----------
		 *
		 * index: integer
		 *		Positive index
		 *
		 * Returns
		 * -------
		 *
		 * value: integer
		 *		Size of Extent at dimension ``index``
		 *
		 * \endrst
		 */
		LR_INLINE const lr_int &operator[](size_t index) const
		{
			if (index >= m_dims)
				throw std::out_of_range("Index " + std::to_string(index)
										+ " is out of range for Extent with "
										+ std::to_string(m_dims) + " dimensions");

			return m_extent[index];
		}

		LR_INLINE lr_int &operator[](size_t index)
		{
			if (index >= m_dims)
				throw std::out_of_range("Index " + std::to_string(index)
										+ " is out of range for Extent with "
										+ std::to_string(m_dims) + " dimensions");

			return m_extent[index];
		}

		/**
		 * \rst
		 *
		 * Provided with a target number of elements, attempt to resolve any
		 * automatic dimensions and return a new Extent object that fits these
		 * conditions.
		 *
		 * Parameters
		 * ----------
		 *
		 * target: integer
		 *		Positive integer number of elements
		 *
		 * Returns
		 * -------
		 *
		 * fixed: ``Extent``
		 *		An Extent object with any automatic dimensions resolved to fit the
		 *		target number of elements
		 * \endrst
		 */
		LR_INLINE Extent fixed(size_t target) const
		{
			size_t neg = 0;
			for (size_t i = 0; i < m_dims; ++i)
				if (m_extent[i] < 0) ++neg;

			if (neg > 1)
				throw std::invalid_argument("Cannot construct Extent with more than"
											" one automatic values. " +
											std::to_string(neg) +
											" automatic values were found.");

			// If no automatic dimensions exist, quick return
			if (!m_containsAutomatic)
				return *this;

			size_t autoIndex = 0;
			size_t prod = 1;

			for (size_t i = 0; i < m_dims; ++i)
			{
				if (m_extent[i] < 1)
					autoIndex = i;
				else
					prod *= m_extent[i];
			}

			if (target % prod == 0)
			{
				Extent res = *this;
				res.m_extent[autoIndex] = target / prod;
				return res;
			}

			throw std::runtime_error("Could not resolve automatic dimension of "
									 " Extent to fit " + std::to_string(target)
									 + " elements");
		}

		/**
		 * \rst
		 *
		 * Returns true if two extents are identical. This involves number of
		 * dimensions, automatic dimensions, and the sizes of each dimension in the
		 * Extents.
		 *
		 * \endrst
		 */
		LR_INLINE bool operator==(const Extent &other) const
		{
			if (m_dims != other.m_dims)
				return false;

			if (m_containsAutomatic != other.m_containsAutomatic)
				return false;

			for (size_t i = 0; i < m_dims; ++i)
			{
				if (m_extent[i] != other.m_extent[i])
					return false;
			}

			return true;
		}

		/**
		 * \rst
		 *
		 * Returns true if two Extents are *not* equal.
		 *
		 * See ``librapid::Extent::operator==``
		 *
		 * \endrst
		 */
		LR_INLINE bool operator!=(const Extent &other) const
		{
			return !(*this != other);
		}

		/**
		 * \rst
		 *
		 * Reorder the contents of the Extent object.
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
			Extent temp = *this;

			for (size_t i = 0; i < order.size(); ++i)
				m_extent[i] = temp.m_extent[order[i]];
		}

		/**
		 * \rst
		 *
		 * Generate a string representation of the Extent. The result takes the
		 * following general form:
		 *
		 * :math:`\text{Extent}(\text{dim}_0, \text{dim}_1, ... , \text{dim}_{n-1})`
		 *
		 * \endrst
		 */
		LR_INLINE std::string str() const
		{
			std::stringstream res;
			res << "Extent(";
			for (size_t i = 0; i < m_dims; ++i)
			{
				if (m_extent[i] == librapid::AUTO)
					res << "librapid::AUTO";
				else
					res << m_extent[i];

				if (i < m_dims - 1)
					res << ", ";
			}
			res << ")";

			return res.str();
		}

		LR_INLINE ESIterator begin() const
		{
			return ESIterator((lr_int *) m_extent);
		}

		LR_INLINE ESIterator end() const
		{
			return ESIterator((lr_int *) m_extent + m_dims);
		}

	private:
		LR_INLINE void update()
		{
			size_t neg = 0;
			m_size = 1;
			for (size_t i = 0; i < m_dims; i++)
			{
				m_size *= m_extent[i];

				if (m_extent[i] < 0)
				{
					neg++;
					m_extent[i] = AUTO;
				}
			}

			if (neg == 1)
				m_containsAutomatic = true;
			else if (neg > 1)
				throw std::invalid_argument("Cannot construct Extent with more than"
											" one automatic values. " +
											std::to_string(neg) +
											" automatic values were found.");
			else
				m_containsAutomatic = false;
		}

	private:
		lr_int m_extent[LIBRAPID_MAX_DIMS]{};
		size_t m_dims = 0;
		bool m_containsAutomatic = false;
		lr_int m_size = 0;
	};

	LR_INLINE std::ostream &operator<<(std::ostream &os, const Extent &extent)
	{
		return os << extent.str();
	}
}

#endif // LIBRAPID_EXTENT