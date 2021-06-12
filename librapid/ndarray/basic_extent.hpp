#ifndef NDARRAY_BASIC_EXTENTS
#define NDARRAY_BASIC_EXTENTS

#include <librapid/ndarray/utils.hpp>
#include <librapid/math/rapid_math.hpp>

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

		template<typename T>
		class extent_iterator
		{
		public:
			using iterator_category = std::random_access_iterator_tag;
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using pointer = value_type *;
			using reference = value_type &;

			extent_iterator() = default;

			extent_iterator(pointer start)
			{
				m_ptr = start;
			}

			~extent_iterator() = default;

			ND_INLINE extent_iterator<T> &operator=(const extent_iterator<T> &other) = default;

			ND_INLINE extent_iterator<T> &operator=(pointer ptr)
			{
				m_ptr = ptr;
				return *this;
			}

			ND_INLINE operator bool() const
			{
				return m_ptr ? true : false;
			}

			ND_INLINE bool operator==(const extent_iterator<T> &other) const
			{
				return m_ptr == other.get_const_ptr();
			}

			ND_INLINE bool operator!=(const extent_iterator<T> &other) const
			{
				return m_ptr != other.get_const_ptr();
			}

			ND_INLINE extent_iterator<T> &operator+=(const difference_type &movement)
			{
				m_ptr += movement;
				return (*this);
			}

			ND_INLINE extent_iterator<T> &operator-=(const difference_type &movement)
			{
				m_ptr -= movement;
				return (*this);
			}

			ND_INLINE extent_iterator<T> &operator++()
			{
				++m_ptr;
				return (*this);
			}

			ND_INLINE extent_iterator<T> &operator--()
			{
				--m_ptr;
				return (*this);
			}

			ND_INLINE extent_iterator<T> operator++(int)
			{
				auto temp(*this);
				++m_ptr;
				return temp;
			}

			ND_INLINE extent_iterator<T> operator--(int)
			{
				auto temp(*this);
				--m_ptr;
				return temp;
			}

			ND_INLINE extent_iterator<T> operator+(const difference_type &movement) const
			{
				return extent_iterator<T>(m_ptr + movement);
			}

			ND_INLINE extent_iterator<T> operator-(const difference_type &movement) const
			{
				return extent_iterator<T>(m_ptr - movement);
			}

			ND_INLINE difference_type operator-(const extent_iterator<T> &raw_iterator) const
			{
				return std::distance(raw_iterator.get_ptr(), get_ptr());
			}

			ND_INLINE T &operator*()
			{
				return *m_ptr;
			}

			ND_INLINE const T &operator*() const
			{
				return *m_ptr;
			}

			ND_INLINE T *operator->()
			{
				return m_ptr;
			}

			ND_INLINE const pointer get_const_ptr() const
			{
				return m_ptr;
			}

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
			basic_extent(PTR *data, nd_int dims)
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