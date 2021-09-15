#ifndef LIBRAPID_EXTENT
#define LIBRAPID_EXTENT

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/iterators.hpp>
#include <iostream>
#include <sstream>
#include <ostream>
#include <vector>

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
		inline Extent() = default;

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
		Extent(const std::initializer_list<lr_int> &data);
		Extent(const std::vector<lr_int> &data);

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
		Extent(size_t dims);

		Extent(const Extent &other);

	#ifdef LIBRAPID_PYTHON
		Extent(py::args args);
	#endif // LIBRAPID_PYTHON

		Extent &operator=(const Extent &other);

		/**
		 * \rst
		 *
		 * Get the number of dimensions of the Extent
		 *
		 * \endrst
		 */
		inline const size_t &ndim() const
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
		inline lr_int size() const
		{
			return m_size;
		}

		/**
		 * \rst
		 *
		 * Return a pointer to the raw data of this extent
		 *
		 * \endrst
		 */
		inline const int64_t *__restrict raw() const
		{
			return m_extent;
		}

		/**
		* \rst
		*
		* Convert the Stride object to an std::vector and return the result
		*
		* \endrst
		*/
		inline std::vector<lr_int> toVec() const
		{
			std::vector<lr_int> res(m_dims);
			for (size_t i = 0; i < m_dims; ++i)
				res[i] = m_extent[i];
			return res;
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
		inline bool containsAutomatic() const
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
		const lr_int &operator[](size_t index) const;

		lr_int &operator[](size_t index);

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
		Extent fixed(size_t target) const;

		/**
		 * \rst
		 *
		 * Returns true if two extents are identical. This involves number of
		 * dimensions, automatic dimensions, and the sizes of each dimension in the
		 * Extents.
		 *
		 * \endrst
		 */
		bool operator==(const Extent &other) const;

		/**
		 * \rst
		 *
		 * Returns true if two Extents are *not* equal.
		 *
		 * See ``librapid::Extent::operator==``
		 *
		 * \endrst
		 */
		inline bool operator!=(const Extent &other) const
		{
			return !(*this == other);
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
		void reorder(const std::vector<size_t> &order);
		void reorder(const std::vector<lr_int> &order);

		Extent subExtent(lr_int start = -1, lr_int end = -1) const;

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
		std::string str() const;

		inline ESIterator begin() const
		{
			return ESIterator((lr_int *) m_extent);
		}

		inline ESIterator end() const
		{
			return ESIterator((lr_int *) m_extent + m_dims);
		}

	private:
		void update();

	private:
		lr_int m_extent[LIBRAPID_MAX_DIMS]{};
		size_t m_dims = 0;
		bool m_containsAutomatic = false;
		lr_int m_size = 0;
	};

	inline std::ostream &operator<<(std::ostream &os, const Extent &extent)
	{
		return os << extent.str();
	}
}

#endif // LIBRAPID_EXTENT