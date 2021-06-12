#ifndef NDARRAY_BASIC_STRIDE
#define NDARRAY_BASIC_STRIDE

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
		class basic_stride;

		/// <summary>
		/// The stride iterator class allows you to iterate
		/// over the stride of an array, accessing the values
		/// one by one in order
		/// </summary>
		/// <typeparam name="T">The datatype of the extestrident being iterated over</typeparam>
		template<typename T>
		class stride_iterator
		{
		public:
			using iterator_category = std::random_access_iterator_tag;
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using pointer = value_type *;
			using reference = value_type &;

			/// <summary>
			/// Create an empty stride iterator
			/// </summary>
			stride_iterator() = default;

			/// <summary>
			/// Create a new stride iterator using
			/// the starting pointer
			/// </summary>
			/// <param name="start">The starting memory location</param>
			stride_iterator(pointer start)
			{
				m_ptr = start;
			}

			~stride_iterator() = default;

			/// <summary>
			/// Set one stride iterator equal to another
			/// </summary>
			/// <param name="other">Another stride iterator</param>
			/// <returns>A reference to the original iterator</returns>
			ND_INLINE stride_iterator<T> &operator=(const stride_iterator<T> &other) = default;

			/// <summary>
			/// Set an stride iterator to a pointer
			/// of the same type
			/// </summary>
			/// <param name="ptr">The pointer to iterate</param>
			/// <returns>Reference to original iterator</returns>
			ND_INLINE stride_iterator<T> &operator=(pointer ptr)
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
			/// Check whether one stride iterator is equal to another
			/// stride iterator of the same type
			/// </summary>
			/// <param name="other">The iterator to compare against</param>
			/// <returns>True if the iterators are equal</returns>
			ND_INLINE bool operator==(const stride_iterator<T> &other) const
			{
				return m_ptr == other.get_const_ptr();
			}

			/// <summary>
			/// Check for a lack of equality with another iterator
			/// of the same type.
			/// </summary>
			/// <param name="other">The iterator to compare agianst</param>
			/// <returns>True if the iterators are not equal</returns>
			ND_INLINE bool operator!=(const stride_iterator<T> &other) const
			{
				return m_ptr != other.get_const_ptr();
			}

			/// <summary>
			/// Increment the pointer by a specific offset
			/// </summary>
			/// <param name="movement">The offset for the pointer</param>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE stride_iterator<T> &operator+=(const difference_type &movement)
			{
				m_ptr += movement;
				return (*this);
			}

			/// <summary>
			/// Decrement the pointer by a specific offset
			/// </summary>
			/// <param name="movement">The offset for the pointer</param>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE stride_iterator<T> &operator-=(const difference_type &movement)
			{
				m_ptr -= movement;
				return (*this);
			}

			/// <summary>
			/// Increment the pointer by one
			/// </summary>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE stride_iterator<T> &operator++()
			{
				++m_ptr;
				return (*this);
			}

			/// <summary>
			/// Decrement the iterator by one
			/// </summary>
			/// <returns>Reference to the iterator</returns>
			ND_INLINE stride_iterator<T> &operator--()
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
			ND_INLINE stride_iterator<T> operator++(int)
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
			ND_INLINE stride_iterator<T> operator--(int)
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
			ND_INLINE stride_iterator<T> operator+(const difference_type &movement) const
			{
				return stride_iterator<T>(m_ptr + movement);
			}

			/// <summary>
			/// Return a new iterator decremented
			/// with a given offset
			/// </summary>
			/// <param name="movement"></param>
			/// <returns>A new iterator</returns>
			ND_INLINE stride_iterator<T> operator-(const difference_type &movement) const
			{
				return stride_iterator<T>(m_ptr - movement);
			}

			/// <summary>
			/// Find the difference between one iterator
			/// and another
			/// </summary>
			/// <param name="raw_iterator"></param>
			/// <returns>The difference between the iterators</returns>
			ND_INLINE difference_type operator-(const stride_iterator<T> &raw_iterator) const
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
		class basic_stride
		{
		public:
			basic_stride() = default;

			basic_stride(const std::initializer_list<T> &vals) :
				basic_stride(std::vector<T>(vals.begin(), vals.end()))
			{}

			basic_stride(nd_int n)
			{
				m_is_trivial = true;
				m_dims = n;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < n; i++)
				{
					m_stride[i] = 1;
					m_stride_alt[i] = 1;
				}
			}

			basic_stride(const basic_stride<T> &o)
			{
				m_dims = o.m_dims;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				memcpy(m_stride, o.m_stride, sizeof(T) * m_dims);
				memcpy(m_stride_alt, o.m_stride_alt, sizeof(T) * m_dims);

				m_is_trivial = check_trivial();
			}

			template<typename V>
			basic_stride(const std::vector<V> &strides)
			{
				m_dims = strides.size();

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_stride[i] = strides[i];
					m_stride_alt[i] = strides[m_dims - i - 1];
				}

				m_is_trivial = check_trivial();
			}

			template<typename A, typename B>
			basic_stride(const std::pair<A, B> &pair)
			{
				m_dims = pair.second;

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_stride[i] = pair.first[i];
					m_stride_alt[i] = pair.first[m_dims - i - 1];
				}

				m_is_trivial = check_trivial();
			}

			template<typename PTR>
			basic_stride(PTR data, nd_int dims)
			{
				m_dims = dims;

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_stride[i] = data[i];
					m_stride_alt[i] = data[m_dims - i - 1];
				}

				m_is_trivial = check_trivial();
			}

		#if LIBRAPID_BUILD == 1

			// PyBind11 specific constructor

			basic_stride(py::args args)
			{
				m_dims = py::len(args);

				if (m_dims > ND_MAX_DIMS)
				{
					m_dims = ND_MAX_DIMS + 1;
					return;
				}

				for (nd_int i = 0; i < m_dims; i++)
				{
					m_stride[i] = py::cast<nd_int>(args[i]);
					m_stride_alt[i] = py::cast<nd_int>(args[m_dims - i - 1]);
				}

				m_is_trivial = check_trivial();
			}

		#endif

			template<typename V>
			static basic_stride<T> from_extent(const std::vector<V> &extent)
			{
				return from_extent(extent.data(), extent.size());
			}

			template<typename V>
			static basic_stride<T> from_extent(const V *extent, nd_int dims)
			{
				if (math::anyBelow(extent, dims, 1))
					throw std::domain_error("basic_stride cannot contain values less than 1");

				basic_stride<T> res;

				res.m_dims = dims;

				V prod = 1;
				for (nd_int i = 0; i < dims; i++)
				{
					res.m_stride[dims - i - 1] = (T) prod;
					prod *= extent[dims - i - 1];
				}

				for (nd_int i = 0; i < dims; i++)
					res.m_stride_alt[i] = res.m_stride[dims - i - 1];

				res.m_is_trivial = true;

				return res;
			}

			ND_INLINE basic_stride &operator=(const basic_stride<T> &o)
			{
				if (this == &o)
					return *this;

				m_dims = o.m_dims;

				m_is_trivial = o.m_is_trivial;

				memcpy(m_stride, o.m_stride, sizeof(T) * m_dims);
				memcpy(m_stride_alt, o.m_stride_alt, sizeof(T) * m_dims);

				return *this;
			}

			ND_INLINE bool operator==(const basic_stride<T> &other) const
			{
				return utils::check_ptr_match(m_stride, m_dims, other.m_stride, other.m_dims);
			}

			template<typename I>
			ND_INLINE T &operator[](I index)
			{
				return m_stride[index];
			}

			template<typename I>
			ND_INLINE const T &operator[](I index) const
			{
				return m_stride[index];
			}

			ND_INLINE nd_int ndim() const
			{
				return m_dims;
			}

			ND_INLINE bool is_valid() const
			{
				return m_dims > 0;
			}

			ND_INLINE const auto &get_stride() const
			{
				return m_stride;
			}

			ND_INLINE const auto &get_stride_alt() const
			{
				return m_stride_alt;
			}

			ND_INLINE const bool is_trivial() const
			{
				return m_is_trivial;
			}

			ND_INLINE void set_dimensions(nd_int new_dims)
			{
				m_dims = new_dims;
			}

			template<typename O>
			ND_INLINE void reshape(const std::vector<O> &order)
			{
				// No validation. This should be completed by the caller of this function
				T new_stride[ND_MAX_DIMS]{};
				T new_stride_alt[ND_MAX_DIMS]{};

				nd_int i = 0;
				for (const auto &index : order)
				{
					new_stride[index] = m_stride[i];
					new_stride_alt[index] = m_stride_alt[i];
					++i;
				}

				memcpy(m_stride, new_stride, sizeof(T) * m_dims);
				memcpy(m_stride_alt, new_stride_alt, sizeof(T) * m_dims);

				m_is_trivial = check_trivial();
			}

			/// <summary>
			/// The begin function returns an iterator
			/// pointing to the first element of the
			/// stride, allowing you to iterate over
			/// it easily
			/// </summary>
			/// <returns>Stride iterator object</returns>
			ND_INLINE stride_iterator<T> begin() const
			{
				return stride_iterator<T>((T *) m_stride);
			}

			/// <summary>
			/// The end function returns an iterator
			/// pointing to the memory location one
			/// element beyond the end of the stride,
			/// allowing you to iterate over it
			/// </summary>
			/// <returns>Stride iterator object</returns>
			ND_INLINE stride_iterator<T> end() const
			{
				return stride_iterator<T>((T *) m_stride + m_dims);
			}

			ND_INLINE std::string str() const
			{
				auto stream = std::stringstream();
				for (nd_int i = 0; i < m_dims; i++)
				{
					if (i == m_dims - 1) stream << m_stride[i];
					else stream << m_stride[i] << ", ";
				}
				return "stride(" + stream.str() + ")";
			}

		private:

			ND_INLINE const bool check_trivial() const
			{
				if (m_dims == 1)
					return m_stride[0] == 1;

				for (nd_int i = 0; i < m_dims - 1; i++)
				{
					if (m_stride[i] < m_stride[i + 1])
						goto not_trivial;
				}
				return true;
			not_trivial:
				return false;
			}

		private:
			T m_stride[ND_MAX_DIMS]{};
			T m_stride_alt[ND_MAX_DIMS]{};

			nd_int m_dims = 0;
			bool m_is_trivial = false;
		};

		using stride = basic_stride<long long>;

		template<typename T>
		std::ostream &operator<<(std::ostream &os, const basic_stride<T> &s)
		{
			return os << s.str();
		}
	}
}

#endif // NDARRAY_BASIC_STRIDE