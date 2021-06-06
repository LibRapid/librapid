#ifndef NDARRAY_BASIC_EXTENTS
#define NDARRAY_BASIC_EXTENTS

#include <librapid/ndarray/utils.hpp>

#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include <ostream>
#include <vector>

#if LIBRAPID_BUILD == 1
namespace py = pybind11;
#endif

namespace librapid
{
	namespace ndarray
	{
		template<typename T = nd_int, typename std::enable_if<std::is_integral<T>::value, int>::type = 0 >
		class basic_extent;

		/// <summary>
		/// The extent iterator class allows you to iterate
		/// over the extent of an array, accessing the values
		/// one by one in order
		/// </summary>
		/// <typeparam name="T">The datatype of the extent being iterated over</typeparam>
		template<typename T>
		class extent_iterator
		{
		public:
			using iterator_category = std::random_access_iterator_tag;
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using pointer = value_type *;
			using reference = value_type &;

			/// <summary>
			/// Create an empty extent iterator
			/// </summary>
			extent_iterator() = default;

			/// <summary>
			/// Create a new extent iterator using
			/// the starting pointer
			/// </summary>
			/// <param name="start">The starting memory location</param>
			extent_iterator(pointer start)
			{
				m_ptr = start;
			}

			~extent_iterator() = default;

			/// <summary>
			/// Set one extent iterator equal to another
			/// </summary>
			/// <param name="other">Another extent iterator</param>
			/// <returns>A reference to the original iterator</returns>
			ND_INLINE extent_iterator<T> &operator=(const extent_iterator<T> &other) = default;

			/// <summary>
			/// Set an extent iterator to a pointer
			/// of the same type
			/// </summary>
			/// <param name="ptr">The pointer to iterate</param>
			/// <returns>Reference to original iterator</returns>
			ND_INLINE extent_iterator<T> &operator=(pointer ptr)
			{
				m_ptr = ptr;
				return *this;
			}

			/// <summary>
			/// Convert the iterator to a boolean value.
			/// The result is "true" if the pointer is a
			/// valid memory location, otherwise the result
			/// is "false"
			/// </summary>
			/// <returns>True if the iterator's pointer is valid</returns>
			ND_INLINE operator bool() const
			{
				return m_ptr ? true : false;
			}

			/// <summary>
			/// Check whether one extent iterator is equal to another
			/// extent iterator of the same type
			/// </summary>
			/// <param name="other">The iterator to compare against</param>
			/// <returns>True if the iterators are equal</returns>
			ND_INLINE bool operator==(const extent_iterator<T> &other) const
			{
				return m_ptr == other.get_const_ptr();
			}

			/// <summary>
			/// Check for a lack of equality with another iterator
			/// of the same type.
			/// </summary>
			/// <param name="other">The iterator to compare agianst</param>
			/// <returns>True if the iterators are not equal</returns>
			ND_INLINE bool operator!=(const extent_iterator<T> &other) const
			{
				return m_ptr != other.get_const_ptr();
			}

			/// <summary>
			/// Increment the pointer by a specific offset
			/// </summary>
			/// <param name="movement">The offset for the pointer</param>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE extent_iterator<T> &operator+=(const difference_type &movement)
			{
				m_ptr += movement;
				return (*this);
			}

			/// <summary>
			/// Decrement the pointer by a specific offset
			/// </summary>
			/// <param name="movement">The offset for the pointer</param>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE extent_iterator<T> &operator-=(const difference_type &movement)
			{
				m_ptr -= movement;
				return (*this);
			}

			/// <summary>
			/// Increment the pointer by one
			/// </summary>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE extent_iterator<T> &operator++()
			{
				++m_ptr;
				return (*this);
			}

			/// <summary>
			/// Decrement the iterator by one
			/// </summary>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE extent_iterator<T> &operator--()
			{
				--m_ptr;
				return (*this);
			}

			/// <summary>
			/// Increment the iterator, but return
			/// it's unincremented value. For example,
			/// an iterator at memory location zero
			/// would return another iterator at zero,
			/// though the original iterator would then
			/// be at memory location one
			/// </summary>
			/// <param name=""></param>
			/// <returns></returns>
			ND_INLINE extent_iterator<T> operator++(int)
			{
				auto temp(*this);
				++m_ptr;
				return temp;
			}

			/// <summary>
			/// Decrement the iterator, but return
			/// it's original value. For example,
			/// an iterator at memory location one
			/// would return another iterator at one,
			/// though the original iterator would then
			/// be at memory location zero
			/// </summary>
			/// <param name=""></param>
			/// <returns></returns>
			ND_INLINE extent_iterator<T> operator--(int)
			{
				auto temp(*this);
				--m_ptr;
				return temp;
			}

			/// <summary>
			/// Return a new iterator incremented
			/// with a given offset
			/// </summary>
			/// <param name="movement"></param>
			/// <returns>A new iterator</returns>
			ND_INLINE extent_iterator<T> operator+(const difference_type &movement) const
			{
				return extent_iterator<T>(m_ptr + movement);
			}

			/// <summary>
			/// Return a new iterator decremented
			/// with a given offset
			/// </summary>
			/// <param name="movement"></param>
			/// <returns>A new iterator</returns>
			ND_INLINE extent_iterator<T> operator-(const difference_type &movement) const
			{
				return extent_iterator<T>(m_ptr - movement);
			}

			/// <summary>
			/// Find the difference between one iterator
			/// and another
			/// </summary>
			/// <param name="raw_iterator"></param>
			/// <returns>The difference between the iterators</returns>
			ND_INLINE difference_type operator-(const extent_iterator<T> &raw_iterator) const
			{
				return std::distance(raw_iterator.get_ptr(), get_ptr());
			}

			/// <summary>
			/// Dereference the iterator and return
			/// a reference to the value at the current
			/// point in the iterator
			/// </summary>
			/// <returns>A value reference</returns>
			ND_INLINE T &operator*()
			{
				return *m_ptr;
			}

			/// <summary>
			/// Dereference the iterator and return
			/// a const reference to the value at the
			/// current point in the iterator
			/// </summary>
			/// <returns>A const value reference</returns>
			ND_INLINE const T &operator*() const
			{
				return *m_ptr;
			}

			/// <summary>
			/// Access the pointer
			/// </summary>
			/// <returns></returns>
			ND_INLINE T *operator->()
			{
				return m_ptr;
			}

			/// <summary>
			/// Access the const pointer of the iterator
			/// </summary>
			/// <returns>A const pointer</returns>
			ND_INLINE const pointer get_const_ptr() const
			{
				return m_ptr;
			}

			/// <summary>
			/// Access the non-const pointer of the iterator
			/// </summary>
			/// <returns>A pointer</returns>
			ND_INLINE pointer get_ptr() const
			{
				return m_ptr;
			}

		private:
			pointer m_ptr = nullptr;
		};

		template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
		class basic_extent
		{
		public:
			basic_extent() = default;

			template<typename V>
			basic_extent(const std::initializer_list<V> &vals)
				: basic_extent(std::vector<V>(vals.begin(), vals.end()))
			{}

			template<typename V>
			basic_extent(const std::vector<V> &vals)
			{
				m_dims = vals.size();

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_extent[i] = vals[i];
					m_extent_alt[i] = vals[m_dims - i - 1];
				}

				if (math::anyBelow(m_extent, m_dims, 1))
					throw std::domain_error("basic_extent cannot contain values less than 1");
			}

			basic_extent(nd_int n)
			{
				m_dims = n;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_extent[i] = 1;
					m_extent_alt[i] = 1;
				}
			}

			basic_extent(const basic_extent<T> &o)
			{
				m_dims = o.m_dims;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				memcpy(m_extent, o.m_extent, sizeof(T) * m_dims);
				memcpy(m_extent_alt, o.m_extent_alt, sizeof(T) * m_dims);
			}

			template<typename A, typename B>
			basic_extent(const std::pair<A, B> &pair)
			{
				m_dims = pair.second;

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_extent[i] = pair.first[i];
					m_extent_alt[i] = pair.first[m_dims - i - 1];
				}
			}

			template<typename PTR>
			basic_extent(PTR data, nd_int dims)
			{
				m_dims = dims;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_extent[i] = data[i];
					m_extent_alt[i] = data[m_dims - i - 1];
				}
			}

		#if LIBRAPID_BUILD == 1

			// PyBind11 specific constructor

			basic_extent(py::args args)
			{
				m_dims = py::len(args);

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_extent[i] = py::cast<nd_int>(args[i]);
					m_extent_alt[i] = py::cast<nd_int>(args[m_dims - i - 1]);
				}

				if (math::anyBelow(m_extent, m_dims, 1))
					throw std::domain_error("basic_extent cannot contain values less than 1");
			}

		#endif

			ND_INLINE basic_extent &operator=(const basic_extent<T> &o)
			{
				if (this == &o)
					return *this;

				m_dims = o.m_dims;
				memcpy(m_extent, o.m_extent, sizeof(T) * m_dims);
				memcpy(m_extent_alt, o.m_extent_alt, sizeof(T) * m_dims);
				return *this;
			}

			ND_INLINE T &operator[](nd_int index)
			{
				if (index < 0 || index >= m_dims)
					throw std::out_of_range("Index " + std::to_string(index)
											+ " is out of range for extent with "
											+ std::to_string(m_dims) + " dimensions");

				return m_extent[index];
			}

			ND_INLINE const T &operator[](nd_int index) const
			{
				if (index < 0 || index >= m_dims)
					throw std::out_of_range("Index " + std::to_string(index)
											+ " is out of range for extent with "
											+ std::to_string(m_dims) + " dimensions");

				return m_extent[index];
			}

			ND_INLINE T &operator()(nd_int index, bool normal)
			{
				if (normal)
					return m_extent[index];
				return m_extent_alt[index];
			}

			ND_INLINE const T &operator()(nd_int index, bool normal) const
			{
				if (normal)
					return m_extent[index];
				return m_extent_alt[index];
			}

			ND_INLINE basic_extent<T> compressed() const
			{
				if (math::product(m_extent, m_dims) == 1)
					return basic_extent({1});

				std::vector<T> res;
				for (nd_int i = 0; i < m_dims; i++)
					if (m_extent[i] != 1)
						res.emplace_back(m_extent[i]);

				return basic_extent(res);
			}

			ND_INLINE nd_int ndim() const
			{
				return m_dims;
			}

			ND_INLINE bool is_valid() const
			{
				return m_dims > 0 && m_dims < ND_MAX_DIMS;
			}

			ND_INLINE const auto &get_extent() const
			{
				return m_extent;
			}

			ND_INLINE const auto &get_extent_alt() const
			{
				return m_extent_alt;
			}

			ND_INLINE bool operator==(const basic_extent<T> &test) const
			{
				return utils::check_ptr_match(m_extent, m_dims, test.m_extent, test.m_dims);
			}

			template<typename O, typename std::enable_if<std::is_integral<O>::value, int>::type = 0>
			ND_INLINE void reshape(const std::vector<O> &order)
			{
				// No validation. This should be completed by the caller of this function
				nd_int size = m_dims;

				nd_int new_extent[ND_MAX_DIMS]{};
				nd_int new_extent_alt[ND_MAX_DIMS]{};

				nd_int i = 0;
				for (const auto &index : order)
				{
					new_extent[index] = m_extent[i];
					new_extent_alt[index] = m_extent_alt[i];
					++i;
				}

				memcpy(m_extent, new_extent, sizeof(nd_int) * size);
				memcpy(m_extent_alt, new_extent_alt, sizeof(nd_int) * size);
			}

			ND_INLINE extent_iterator<T> begin() const
			{
				return extent_iterator<T>((T *) m_extent);
			}

			ND_INLINE extent_iterator<T> end() const
			{
				return extent_iterator<T>((T *) m_extent + m_dims);
			}

			ND_INLINE std::string str() const
			{
				auto stream = std::stringstream();
				for (nd_int i = 0; i < m_dims; i++)
				{
					if (i == m_dims - 1) stream << m_extent[i];
					else stream << m_extent[i] << ", ";
				}
				return "extent(" + stream.str() + ")";
			}

		private:
			T m_extent[ND_MAX_DIMS]{};
			T m_extent_alt[ND_MAX_DIMS]{};

			nd_int m_dims = 0;
		};

		using extent = basic_extent<nd_int>;

		template<typename T>
		std::ostream &operator<<(std::ostream &os, const basic_extent<T> &s)
		{
			return os << s.str();
		}
	}
}

#endif // NDARRAY_BASIC_EXTENTS