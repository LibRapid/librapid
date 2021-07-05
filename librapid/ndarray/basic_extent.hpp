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
	template<typename T = lr_int, typename std::enable_if<std::is_integral<T>::value, int>::type = 0 >
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

		LR_INLINE extent_iterator<T> &operator=(const extent_iterator<T> &other) = default;

		LR_INLINE extent_iterator<T> &operator=(pointer ptr)
		{
			m_ptr = ptr;
			return *this;
		}

		LR_INLINE operator bool() const
		{
			return m_ptr ? true : false;
		}

		LR_INLINE bool operator==(const extent_iterator<T> &other) const
		{
			return m_ptr == other.get_const_ptr();
		}

		LR_INLINE bool operator!=(const extent_iterator<T> &other) const
		{
			return m_ptr != other.get_const_ptr();
		}

		LR_INLINE extent_iterator<T> &operator+=(const difference_type &movement)
		{
			m_ptr += movement;
			return (*this);
		}

		LR_INLINE extent_iterator<T> &operator-=(const difference_type &movement)
		{
			m_ptr -= movement;
			return (*this);
		}

		LR_INLINE extent_iterator<T> &operator++()
		{
			++m_ptr;
			return (*this);
		}

		LR_INLINE extent_iterator<T> &operator--()
		{
			--m_ptr;
			return (*this);
		}

		LR_INLINE extent_iterator<T> operator++(int)
		{
			auto temp(*this);
			++m_ptr;
			return temp;
		}

		LR_INLINE extent_iterator<T> operator--(int)
		{
			auto temp(*this);
			--m_ptr;
			return temp;
		}

		LR_INLINE extent_iterator<T> operator+(const difference_type &movement) const
		{
			return extent_iterator<T>(m_ptr + movement);
		}

		LR_INLINE extent_iterator<T> operator-(const difference_type &movement) const
		{
			return extent_iterator<T>(m_ptr - movement);
		}

		LR_INLINE difference_type operator-(const extent_iterator<T> &raw_iterator) const
		{
			return std::distance(raw_iterator.get_ptr(), get_ptr());
		}

		LR_INLINE T &operator*()
		{
			return *m_ptr;
		}

		LR_INLINE const T &operator*() const
		{
			return *m_ptr;
		}

		LR_INLINE T *operator->()
		{
			return m_ptr;
		}

		LR_INLINE const pointer get_const_ptr() const
		{
			return m_ptr;
		}

		LR_INLINE pointer get_ptr() const
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

			if (m_dims > LIBRAPID_MAX_DIMS)
			{
				m_dims = LIBRAPID_MAX_DIMS + 1;
				return;
			}

			int neg_1 = 0;

			for (lr_int i = 0; i < m_dims; i++)
			{
				m_extent[i] = vals[i];
				m_extent_alt[i] = vals[m_dims - i - 1];

				if (vals[i] < 0)
				{
					if (vals[i] == (V) -1)
						neg_1++;
					else
						throw std::domain_error("Extent cannot contain a negative number");
				}
			}

			if (neg_1 > 1)
				throw std::domain_error("Extent cannot contain more than 1 automatic dimension");

			if (neg_1 == 1)
				m_contains_automatic = true;
		}

		basic_extent(lr_int n)
		{
			m_dims = n;

			if (m_dims > LIBRAPID_MAX_DIMS)
			{
				m_dims = LIBRAPID_MAX_DIMS + 1;
				return;
			}

			for (lr_int i = 0; i < m_dims; i++)
			{
				m_extent[i] = 1;
				m_extent_alt[i] = 1;
			}
		}

		basic_extent(const basic_extent<T> &o)
		{
			m_dims = o.m_dims;
			m_contains_automatic = o.m_contains_automatic;

			if (m_dims > LIBRAPID_MAX_DIMS)
			{
				m_dims = LIBRAPID_MAX_DIMS + 1;
				return;
			}

			memcpy(m_extent, o.m_extent, sizeof(T) * m_dims);
			memcpy(m_extent_alt, o.m_extent_alt, sizeof(T) * m_dims);
		}

		template<typename A, typename B>
		basic_extent(const std::pair<A, B> &pair)
		{
			m_dims = pair.second;

			for (lr_int i = 0; i < m_dims; i++)
			{
				m_extent[i] = pair.first[i];
				m_extent_alt[i] = pair.first[m_dims - i - 1];
			}
		}

		template<typename PTR>
		basic_extent(PTR *data, lr_int dims)
		{
			m_dims = dims;

			if (m_dims > LIBRAPID_MAX_DIMS)
			{
				m_dims = LIBRAPID_MAX_DIMS + 1;
				return;
			}

			for (lr_int i = 0; i < m_dims; i++)
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

			if (m_dims > LIBRAPID_MAX_DIMS)
			{
				m_dims = LIBRAPID_MAX_DIMS + 1;
				return;
			}

			int neg_1 = 0;

			for (lr_int i = 0; i < m_dims; i++)
			{
				m_extent[i] = py::cast<lr_int>(args[i]);
				m_extent_alt[i] = py::cast<lr_int>(args[m_dims - i - 1]);

				if (m_extent[i] < 0)
				{
					if (m_extent[i] == -1)
						neg_1++;
					else
						throw std::domain_error("Extent cannot contain a negative number");
				}
			}

			if (neg_1 > 1)
				throw std::domain_error("Extent cannot contain more than 1 automatic dimension");

			if (neg_1 == 1)
				m_contains_automatic = true;
		}

	#endif

		LR_INLINE basic_extent &operator=(const basic_extent<T> &o)
		{
			if (this == &o)
				return *this;

			m_dims = o.m_dims;
			memcpy(m_extent, o.m_extent, sizeof(T) * m_dims);
			memcpy(m_extent_alt, o.m_extent_alt, sizeof(T) * m_dims);
			return *this;
		}

		LR_INLINE T &operator[](lr_int index)
		{
			if (index < 0 || index >= m_dims)
				throw std::out_of_range("Index " + std::to_string(index)
										+ " is out of range for extent with "
										+ std::to_string(m_dims) + " dimensions");

			return m_extent[index];
		}

		LR_INLINE const T &operator[](lr_int index) const
		{
			if (index < 0 || index >= m_dims)
				throw std::out_of_range("Index " + std::to_string(index)
										+ " is out of range for extent with "
										+ std::to_string(m_dims) + " dimensions");

			return m_extent[index];
		}

		LR_INLINE T &operator()(lr_int index, bool normal)
		{
			if (normal)
				return m_extent[index];
			return m_extent_alt[index];
		}

		LR_INLINE const T &operator()(lr_int index, bool normal) const
		{
			if (normal)
				return m_extent[index];
			return m_extent_alt[index];
		}

		LR_INLINE basic_extent<T> compressed() const
		{
			if (math::product(m_extent, m_dims) == 1)
				return basic_extent({1});

			std::vector<T> res;
			for (lr_int i = 0; i < m_dims; i++)
				if (m_extent[i] != 1)
					res.emplace_back(m_extent[i]);

			return basic_extent(res);
		}

		LR_INLINE basic_extent<T> fix_automatic(lr_int elems) const
		{
			basic_extent<T> res(m_dims);
			lr_int non_auto_dims = 1;
			lr_int auto_index = -1;
			for (lr_int i = 0; i < m_dims; i++)
			{
				if (m_extent[i] == -1)
				{
					auto_index = i;
				}
				else
				{
					non_auto_dims *= m_extent[i];
					res.m_extent[i] = m_extent[i];
					res.m_extent_alt[i] = m_extent[m_dims - i - 1];
				}
			}

			if (auto_index == -1)
				return *this;

			// Ensure that the number of elements provided is possible
			double check = (double) elems / double(non_auto_dims);
			if (check != (lr_int) check)
				goto invalid;

			res.m_extent[auto_index] = elems / non_auto_dims;
			res.m_extent_alt[m_dims - auto_index - 1] = elems / non_auto_dims;

			return res;

		invalid:
			throw std::domain_error(str() + " cannot be broadcast to " + std::to_string(elems) + " elements");
			return basic_extent();
		}

		LR_INLINE lr_int ndim() const
		{
			return m_dims;
		}

		LR_INLINE bool is_valid() const
		{
			return m_dims > 0 && m_dims < LIBRAPID_MAX_DIMS;
		}

		LR_INLINE bool is_automatic() const
		{
			return m_contains_automatic;
		}

		LR_INLINE const auto &get_extent() const
		{
			return m_extent;
		}

		LR_INLINE const auto &get_extent_alt() const
		{
			return m_extent_alt;
		}

		LR_INLINE bool operator==(const basic_extent<T> &test) const
		{
			return utils::check_ptr_match(m_extent, m_dims, test.m_extent, test.m_dims);
		}

		LR_INLINE bool operator!=(const basic_extent<T> &test) const
		{
			return !(*this == test);
		}

		template<typename O, typename std::enable_if<std::is_integral<O>::value, int>::type = 0>
		LR_INLINE void reshape(const std::vector<O> &order)
		{
			// No validation. This should be completed by the caller of this function
			lr_int size = m_dims;

			lr_int new_extent[LIBRAPID_MAX_DIMS]{};
			lr_int new_extent_alt[LIBRAPID_MAX_DIMS]{};

			lr_int i = 0;
			for (const auto &index : order)
			{
				new_extent[index] = m_extent[i];
				new_extent_alt[index] = m_extent_alt[i];
				++i;
			}

			memcpy(m_extent, new_extent, sizeof(lr_int) * size);
			memcpy(m_extent_alt, new_extent_alt, sizeof(lr_int) * size);
		}

		LR_INLINE extent_iterator<T> begin() const
		{
			return extent_iterator<T>((T *) m_extent);
		}

		LR_INLINE extent_iterator<T> end() const
		{
			return extent_iterator<T>((T *) m_extent + m_dims);
		}

		LR_INLINE std::string str() const
		{
			auto stream = std::stringstream();
			for (lr_int i = 0; i < m_dims; i++)
			{
				if (i == m_dims - 1) stream << m_extent[i];
				else stream << m_extent[i] << ", ";
			}
			return "extent(" + stream.str() + ")";
		}

	private:
		T m_extent[LIBRAPID_MAX_DIMS]{};
		T m_extent_alt[LIBRAPID_MAX_DIMS]{};

		lr_int m_dims = 0;
		bool m_contains_automatic = false;
	};

	using extent = basic_extent<lr_int>;

	template<typename T>
	std::ostream &operator<<(std::ostream &os, const basic_extent<T> &s)
	{
		return os << s.str();
	}
}

#endif // NDARRAY_BASIC_EXTENTS