#ifndef NDARRAY_BASIC_ARRAY
#define NDARRAY_BASIC_ARRAY

#include <memory>
#include <type_traits>

#include <string>
#include <sstream>
#include <ostream>

#include <vector>
#include <algorithm>
#include <atomic>
#include <cmath>

#include <librapid/math/rapid_math.hpp>

// Define this if using a custom cblas interface.
// If it is not defined, the (slower) internal
// interface will be used.
#ifdef LIBRAPID_CBLAS
#include <cblas.h>
#endif

#include <librapid/ndarray/cblas_api.hpp>

// For use in "basic_ndarray::from_data" and "librapid::array"
template<typename T>
using VEC_ = std::vector<T>;

namespace librapid
{
	constexpr lr_int AUTO = -1;

	template<typename T>
	using nd_allocator = std::allocator<T>;

	template<typename T, class alloc = nd_allocator<T>,
		typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
		class basic_ndarray;

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator+(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator-(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator*(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator/(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

	template<typename T, class alloc, typename std::enable_if<std::is_arithmetic<T>::value, int>::type>
	class basic_ndarray
	{
		using _alloc = alloc;

		template<typename A_T, typename B_T, typename B_A>
		friend basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator+(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		friend basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator-(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		friend basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator*(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		friend basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator/(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

	public:

		/**
		 * \rst
		 *
		 * Create an empty n-dimensional array. This array does not have an
		 * extent or a stride, and many functions will not operate correctly
		 * on such an array. For example, printing out an array created with
		 * this function will result in ``[NONE]`` being printed.
		 *
		 * .. Hint::
		 * 		No memory is allocated on the heap when using this function,
		 * 		so it is incredibly fast
		 *
		 * \endrst
		 */
		basic_ndarray()
		{};

		/**
		 * \rst
		 *
		 * Create a new array from a given extent.
		 *
		 * The array created will have the same number of dimensions
		 * as the number of elements passed in the extent object. For
		 * example, passing in ``extent(2, 3)`` will create a 2x3
		 * matrix.
		 *
		 * Parameters
		 * ----------
		 *
		 * size: librapid::extent
		 *		The dimensions of the array
		 *
		 * \endrst
		 */
		template<typename E>
		basic_ndarray(const basic_extent<E> &size) : m_extent(size),
			m_stride(stride::from_extent(size.get_extent(), size.ndim())),
			m_extent_product(math::product(size.get_extent(), size.ndim()))
		{
			if (m_extent.is_automatic())
				throw std::domain_error("Cannot create a new array with an automatic dimension");
			if (m_extent_product < 1)
				return;

			auto state = construct_new();
			m_stride.set_contiguous(true);

			if (state == errors::ALL_OK)
				return;

			if (state == errors::ARRAY_DIMENSIONS_TOO_LARGE)
				throw std::range_error("Too many dimensions in array. Maximum allowed is "
									   + std::to_string(LIBRAPID_MAX_DIMS));
		}

		template<typename E>
		basic_ndarray(const std::vector<E> &size) : basic_ndarray(extent(size))
		{}

		template<typename E>
		basic_ndarray(const std::initializer_list<E> &size) : basic_ndarray(extent(size))
		{}

		/**
		 * \rst
		 *
		 * Creates a new array from a given shape, and fill it with a value.
		 *
		 * For example, creating an array from ``librapid::extent({3, 4}), 5.``
		 * will create the following array:
		 *
		 * .. code-block:: python
		 *
		 * 	[[5. 5. 5. 5.]
		 * 	 [5. 5. 5. 5.]
		 * 	 [5. 5. 5. 5.]]
		 *
		 * Parameters
		 * ----------
		 *
		 * size: librapid::extent
		 * 		The dimensions of the array
		 * val: Any arithmetic type
		 * 		The fill value
		 *
		 * \endrst
		 */
		template<typename E, typename V>
		basic_ndarray(const basic_extent<E> &size, V val) : m_extent(size),
			m_stride(stride::from_extent(size.get_extent(), size.ndim())),
			m_extent_product(math::product(size.get_extent(), size.ndim()))
		{
			if (m_extent_product < 1)
				return;

			if (m_extent.is_automatic())
				throw std::domain_error("Cannot create a new array with an automatic dimension");

			auto state = construct_new();
			m_stride.set_contiguous(true);

			if (state == errors::ALL_OK)
			{
				fill((T) val);
				return;
			}

			if (state == errors::ARRAY_DIMENSIONS_TOO_LARGE)
				throw std::range_error("Too many dimensions in array. Maximum allowed is "
									   + std::to_string(LIBRAPID_MAX_DIMS));
		}

		template<typename E, typename V>
		basic_ndarray(const std::vector<E> &size, V val) : basic_ndarray(extent(size), val)
		{}

		template<typename E, typename V>
		basic_ndarray(const std::initializer_list<E> &size, V val) : basic_ndarray(extent(size), val)
		{}

		/**
		 * \rst
		 *
		 * Create a new array from an existing one, where the new array's data
		 * is linked to the existing one's, so an update in one will update
		 * the data in the other.
		 *
		 * .. Attention::
		 * 		The shape and stride of the arrays will be the same, so a sub-optimal
		 * 		stride in the original array will also result in a sub-optimal stride
		 * 		in the new one. To mitigate this, try using ``.copy()`` instead.
		 *
		 * Parameters
		 * ----------
		 *
		 * arr: librapid::ndarray
		 * 		The array to reference
		 *
		 * \endrst
		 */
		basic_ndarray(const basic_ndarray<T> &arr) : m_extent(arr.m_extent),
			m_stride(arr.m_stride), m_origin_references(arr.m_origin_references),
			m_data_origin(arr.m_data_origin), m_data_start(arr.m_data_start),
			m_extent_product(arr.m_extent_product), m_origin_size(arr.m_origin_size),
			m_is_scalar(arr.m_is_scalar)
		{
			increment();
		}

		LR_INLINE void set_to(const basic_ndarray<T> &other)
		{
			decrement();

			m_data_origin = other.m_data_origin;
			m_origin_references = other.m_origin_references;

			m_origin_size = other.m_origin_size;

			m_data_start = other.m_data_start;

			m_stride = other.m_stride;
			m_extent = other.m_extent;
			m_extent_product = other.m_extent_product;
			m_is_scalar = other.m_is_scalar;

			m_stride.set_contiguous(other.m_stride.is_contiguous());

			increment();
		}

		template<typename V>
		LR_INLINE static basic_ndarray<T> from_data(V scalar)
		{
			basic_ndarray<T> res({1});
			res.m_data_start[0] = (T) scalar;
			res.m_is_scalar = true;
			return res;
		}

		/**
		* \rst
		*
		* Create a new array from the provided data. The function supports
		* creation from a scalar value, as well as vectors, matrices and
		* higher dimensional data.
		*
		* For arrays with dimensions greater than or equal to 2 (i.e. matrices
		* and above) the sub-arrays *must* be the same size, otherwise an error
		* will be thrown.
		*
		* Examples
		* --------
		*
		* .. code-block:: C++
		* 		:caption: C++ Example
		*
		* 		auto my_vec = librapid::ndarray::from_data(VEC_{1, 2, 3, 4});
		* 		std::cout << my_vec << "\n";
		* 		// Prints: [1. 2. 3. 4.]
		*
		* .. code-block:: Python
		* 		:caption: Python Example
		*
		* 		my_vec = librapid.ndarray.from_data([1, 2, 3, 4])
		* 		print(my_vec)
		* 		# Prints: [1. 2. 3. 4.]
		*
		* C++ Specific
		* ------------
		*
		* In C++, the input must be specifically denoted as ``std::vector<...>``
		* for vectors and arrays (this may not be needed when using MSVC). To
		* shorten this process, the typename ``VEC_<...>`` is provided to allow
		* the following: ``librapid:: ... ::from_data(VEC_<VEC_<int>>{{1, 2}, {3, 4}})``.
		*
		* Python Specific
		* ---------------
		*
		* Due to Python's dynamic typing features, a ``list`` or ``tuple`` can be
		* easily converted to an array -- there's no need to worry about any conversions.
		*
		* Parameters
		* ----------
		*
		* data: scalar, vector, nested-vector
		* 		The data for the array
		*
		* Returns
		* -------
		*
		* result: scalar, vector, ndarray
		*		A new array containing the same values that
		*		were originally passed
		*
		* \endrst
		*/
		template<typename V>
		LR_INLINE static basic_ndarray<T> from_data(const std::vector<V> &values)
		{
			// basic_ndarray<T> res(extent({values.size()}));
			basic_ndarray<T> res(utils::extract_size(values));
			for (size_t i = 0; i < values.size(); i++)
				// res.set_value(i, (T) values[i]);
				res[i] = from_data(values[i]);
			return res;
		}

		// template<typename V>
		// LR_INLINE static basic_ndarray<T> from_data(const std::vector<std::vector<V>> &values)
		// {
		// 	basic_ndarray<T> res(extent({values.size()}));
		// 	for (size_t i = 0; i < values.size(); i++)
		// 		res[i] = 
		// 	return res;
		// }

		LR_INLINE basic_ndarray<T> &operator=(const basic_ndarray<T> &arr)
		{
			if (!arr.is_initialized()) // Quick return
				return *this;

			if (!is_initialized())
				construct_new(arr.get_extent(), arr.get_stride());

			if (!(utils::check_ptr_match(m_extent.get_extent(),
				ndim(), arr.m_extent.get_extent(), arr.ndim())))
				throw std::domain_error("Invalid shape for array setting. " +
										m_extent.str() + " and " + arr.get_extent().str() +
										" are not equal.");

			m_extent_product = arr.m_extent_product;

			if (!is_initialized())
			{
				construct_new(arr.m_extent, arr.m_stride);
				m_origin_size = arr.m_origin_size;
				m_is_scalar = arr.m_is_scalar;
			}

			if (m_stride.is_trivial() && arr.m_stride.is_trivial())
			{
				memcpy(m_data_start, arr.m_data_start, m_extent_product * sizeof(T));
			}
			else
			{
				arithmetic::array_op(m_data_start, arr.get_data_start(),
									 m_extent, m_stride, arr.get_stride(), [](T x)
				{
					return x;
				});
			}

			return *this;
		}

		template<typename V>
		LR_INLINE basic_ndarray &operator=(const V &other)
		{
			if (!m_is_scalar)
				throw std::runtime_error("Cannot set non-scalar array with " +
										 m_extent.str() + " to a scalar");

			*m_data_start = T(other);

			return *this;
		}

		~basic_ndarray()
		{
			decrement();
		}

		LR_INLINE lr_int ndim() const
		{
			return (lr_int) m_extent.ndim();
		}

		LR_INLINE lr_int size() const
		{
			return (lr_int) m_extent_product;
		}

		LR_INLINE bool is_initialized() const
		{
			return m_origin_references != nullptr;
		}

		LR_INLINE bool is_scalar() const
		{
			return m_is_scalar;
		}

		LR_INLINE bool is_trivial() const
		{
			return m_stride.is_trivial();
		}

		LR_INLINE bool is_contiguous() const
		{
			return m_stride.is_contiguous();
		}

		LR_INLINE const extent &get_extent() const
		{
			return m_extent;
		}

		LR_INLINE const stride &get_stride() const
		{
			return m_stride;
		}

		LR_INLINE T *get_data_start() const
		{
			return m_data_start;
		}

		template<typename V>
		LR_INLINE bool operator==(const basic_ndarray<V> &arr) const
		{
			if (m_is_scalar)
			{
				if (!arr.m_is_scalar)
					return false;

				return *m_data_start == *arr.get_data_start();
			}

			if (m_extent != arr.get_extent())
				return false;

			if (m_stride.is_trivial() && m_stride.is_contiguous()
				&& arr.get_stride().is_trivial() && arr.get_stride().is_contiguous())
			{
				const V *__restrict tmp = arr.get_data_start();

				for (lr_int i = 0; i < m_extent_product; ++i)
					if (m_data_start[i] != tmp[i])
						return false;

				return true;
			}

			// Non-trivial stride, so use a more complicated accessing
			// method to ensure that the resulting array is contiguous
			// in memory for faster running times overall

			lr_int idim = 0;
			lr_int dims = ndim();

			const auto *__restrict _extent = m_extent.get_extent_alt();
			const auto *__restrict _stride_this = m_stride.get_stride_alt();
			const auto *__restrict _stride_arr = arr.get_stride().get_stride_alt();
			auto *__restrict this_ptr = m_data_start;
			auto *__restrict arr_ptr = arr.get_data_start();

			lr_int coord[LIBRAPID_MAX_DIMS]{};

			do
			{
				if (*this_ptr != *arr_ptr)
					return false;

				for (idim = 0; idim < dims; ++idim)
				{
					if (++coord[idim] == _extent[idim])
					{
						coord[idim] = 0;
						this_ptr = this_ptr - (_extent[idim] - 1) * _stride_this[idim];
						arr_ptr = arr_ptr - (_extent[idim] - 1) * _stride_arr[idim];
					}
					else
					{
						this_ptr = this_ptr + _stride_this[idim];
						arr_ptr = arr_ptr + _stride_arr[idim];
						break;
					}
				}
			} while (idim < dims);

			return true;
		}

		template<typename V>
		LR_INLINE bool operator==(V value) const
		{
			if (!m_is_scalar)
				return false;

			return *m_data_start == value;
		}

		template<typename V>
		LR_INLINE bool operator!=(const basic_ndarray<V> &val) const
		{
			return !(*this == val);
		}

		template<typename V>
		LR_INLINE bool operator!=(const V &val) const
		{
			return !(*this == val);
		}

		LR_INLINE basic_ndarray<T, alloc> operator[](lr_int index)
		{
			using non_const = typename std::remove_const<basic_ndarray<T, alloc>>::type;
			return (non_const) subscript(index);
		}

		LR_INLINE const basic_ndarray<T, alloc> operator[](lr_int index) const
		{
			return subscript(index);
		}

		template<typename I>
		LR_INLINE basic_ndarray<T, alloc> subarray(const std::vector<I> &index) const
		{
			// Validate the index

			if (index.size() != (size_t) ndim())
				throw std::domain_error("Array with " + std::to_string(ndim()) +
										" dimensions requires " + std::to_string(ndim()) +
										" access elements, not " + std::to_string(index.size()));

			lr_int new_shape[LIBRAPID_MAX_DIMS]{};
			lr_int new_stride[LIBRAPID_MAX_DIMS]{};
			lr_int count = 0;

			T *new_start = m_data_start;

			for (size_t i = 0; i < index.size(); i++)
			{
				if (index[i] != AUTO && (index[i] < 0 || index[i] >= m_extent[i]))
				{
					throw std::range_error("Index " + std::to_string(index[i]) +
										   " is out of range for array with extent[" +
										   std::to_string(i) + "] = " + std::to_string(m_extent[i]));
				}

				if (index[i] == AUTO)
				{
					new_shape[count] = m_extent[i];
					new_stride[count] = m_stride[i];
					++count;
				}
				else
					new_start += m_stride[i] * index[i];
			}

			basic_ndarray<T, alloc> res;

			res.m_data_origin = m_data_origin;
			res.m_origin_references = m_origin_references;

			res.m_origin_size = m_origin_size;

			res.m_data_start = new_start;

			res.m_stride = stride(new_stride, count);
			res.m_stride.set_contiguous(res.m_stride.check_contiguous(new_shape, count));

			res.m_extent = extent(new_shape, count);
			res.m_extent_product = math::product(new_shape, count);
			res.m_is_scalar = count == 0;

			increment();

			return res;
		}

		LR_INLINE basic_ndarray<T, alloc> subarray(const std::initializer_list<lr_int> &index) const
		{
			return subarray(std::vector<lr_int>(index.begin(), index.end()));
		}

		template<class F>
		LR_INLINE void fill(const F &filler)
		{
			arithmetic::array_op(m_data_start, m_data_start, m_extent, m_stride, m_stride, [=]<typename V>(V x)
			{
				return filler;
			});
		}

		template<class F>
		LR_INLINE basic_ndarray<T, alloc> filled(const F &filler) const
		{
			basic_ndarray<T, alloc> res;
			res.construct_new(m_extent, m_stride);
			res.fill(filler);

			return res;
		}

		template<typename MIN = T, typename MAX = T>
		LR_INLINE void fill_random(MIN min = 0, MAX max = 1)
		{
			arithmetic::array_op(m_data_start, m_data_start, m_extent, m_stride, m_stride, [=]<typename V>(V x)
			{
				return math::random((T) min, (T) max);
			});
		}

		template<typename MIN = T, typename MAX = T>
		LR_INLINE basic_ndarray<T, alloc> filled_random(MIN min = 0, MAX max = 1) const
		{
			basic_ndarray<T, alloc> res;
			res.construct_new(m_extent, m_stride);
			res.fill_random(min, max);

			return res;
		}

		template<typename LAMBDA>
		LR_INLINE void map(LAMBDA func) const
		{
			if (!m_stride.is_trivial())
			{
				// Non-trivial stride, so use a more complicated accessing
				// method to ensure that the resulting array is contiguous
				// in memory for faster running times overall

				lr_int idim = 0;
				lr_int dims = ndim();

				const auto *__restrict _extent = m_extent.get_extent_alt();
				const auto *__restrict _stride_this = m_stride.get_stride_alt();
				auto *__restrict this_ptr = m_data_start;

				lr_int coord[LIBRAPID_MAX_DIMS]{};

				do
				{
					*this_ptr = func(*this_ptr);

					for (idim = 0; idim < dims; ++idim)
					{
						if (++coord[idim] == _extent[idim])
						{
							coord[idim] = 0;
							this_ptr = this_ptr - (_extent[idim] - 1) * _stride_this[idim];
						}
						else
						{
							this_ptr = this_ptr + _stride_this[idim];
							break;
						}
					}
				} while (idim < dims);
			}
			else
			{
				lr_int end = this->size();
				for (lr_int i = 0; i < end; ++i)
					m_data_start[i] = func(m_data_start[i]);
			}
		}

		template<typename LAMBDA>
		LR_INLINE basic_ndarray<T> mapped(LAMBDA func) const
		{
			auto res = clone();
			res.map(func);
			return res;
		}

		LR_INLINE T to_scalar() const
		{
			if (!m_is_scalar)
				throw std::domain_error("Cannot convert non-scalar array with "
										+ m_extent.str() + " to scalar value");
			return m_data_start[0];
		}

		LR_INLINE basic_ndarray<T> clone() const
		{
			basic_ndarray<T, alloc> res(m_extent);

			res.m_origin_size = m_origin_size;
			res.m_is_scalar = m_is_scalar;

			if (!m_stride.is_trivial())
			{
				// Non-trivial stride, so use a more complicated accessing
				// method to ensure that the resulting array is contiguous
				// in memory for faster running times overall

				lr_int idim = 0;
				lr_int dims = ndim();

				const auto *__restrict _extent = m_extent.get_extent_alt();
				const auto *__restrict _stride_this = m_stride.get_stride_alt();
				auto *__restrict this_ptr = m_data_start;
				auto *__restrict res_ptr = res.get_data_start();

				lr_int coord[LIBRAPID_MAX_DIMS]{};

				do
				{
					*(res_ptr++) = *this_ptr;

					for (idim = 0; idim < dims; ++idim)
					{
						if (++coord[idim] == _extent[idim])
						{
							coord[idim] = 0;
							this_ptr = this_ptr - (_extent[idim] - 1) * _stride_this[idim];
						}
						else
						{
							this_ptr = this_ptr + _stride_this[idim];
							break;
						}
					}
				} while (idim < dims);

				res_ptr -= m_extent_product;
			}
			else
			{
				memcpy(res.m_data_start, m_data_start,
					   m_extent_product * sizeof(T));
			}

			return res;
		}

		void set_value(lr_int index, T val)
		{
			m_data_start[index] = val;
		}

		/**
		 * \rst
		 *
		 * Return :math:`this + other`
		 * See ``librapid::add`` for more details
		 *
		 * \endrst
		 */
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator+(const basic_ndarray<B_T, B_A> &other) const
		{
			return basic_ndarray<T, alloc>::
				array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a + b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`this - other`
		* See ``librapid::sub`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator-(const basic_ndarray<B_T, B_A> &other) const
		{
			return basic_ndarray<T, alloc>::
				array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a - b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`this \times other`
		* See ``librapid::mul`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator*(const basic_ndarray<B_T, B_A> &other) const
		{
			return basic_ndarray<T, alloc>::
				array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a * b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`\frac{this}{other}`
		* See ``librapid::div`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator/(const basic_ndarray<B_T, B_A> &other) const
		{
			return basic_ndarray<T, alloc>::
				array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a / b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`this + other`
		* See ``librapid::add`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator+(const B_T &other) const
		{
			return basic_ndarray<T, alloc>::
				array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a + b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`this - other`
		* See ``librapid::sub`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator-(const B_T &other) const
		{
			return basic_ndarray<T, alloc>::
				array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a - b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`this \times other`
		* See ``librapid::mul`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator*(const B_T &other) const
		{
			return basic_ndarray<T, alloc>::
				array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a * b;
			});
		}

		/**
		* \rst
		*
		* Return :math:`\frac{this}{other}`
		* See ``librapid::div`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			operator/(const B_T &other) const
		{
			return basic_ndarray<T, alloc>::
				array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a / b;
			});
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this + other`
		* See ``librapid::add`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<T> &operator+=(const basic_ndarray<B_T, B_A> &other)
		{
			basic_ndarray<T, alloc>::
				array_array_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a + b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this - other`
		* See ``librapid::sub`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<T> &operator-=(const basic_ndarray<B_T, B_A> &other)
		{
			basic_ndarray<T, alloc>::
				array_array_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a - b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this \times other`
		* See ``librapid::mul`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<T> &operator*=(const basic_ndarray<B_T, B_A> &other)
		{
			basic_ndarray<T, alloc>::
				array_array_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a * b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`\frac{this}{other}`
		* See ``librapid::div`` for more details
		*
		* \endrst
		*/
		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<T> &operator/=(const basic_ndarray<B_T, B_A> &other)
		{
			basic_ndarray<T, alloc>::
				array_array_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a / b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this + other`
		* See ``librapid::add`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<T> &operator+=(const B_T &other)
		{
			basic_ndarray<T, alloc>::
				array_scalar_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a + b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this - other`
		* See ``librapid::sub`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<T> &operator-=(const B_T &other)
		{
			basic_ndarray<T, alloc>::
				array_scalar_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a - b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`this \times other`
		* See ``librapid::mul`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<T> &operator*=(const B_T &other)
		{
			basic_ndarray<T, alloc>::
				array_scalar_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a * b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Set ``this`` equal to :math:`\frac{this}{other}`
		* See ``librapid::div`` for more details
		*
		* \endrst
		*/
		template<typename B_T>
		LR_INLINE basic_ndarray<T> &operator/=(const B_T &other)
		{
			basic_ndarray<T, alloc>::
				array_scalar_arithmetic_inplace(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
			{
				return a / b;
			});
			return *this;
		}

		/**
		* \rst
		*
		* Negate all values in the array and return the
		* result.
		*
		* Example:
		*
		* .. code-block:: python
		*
		*		[1. -2. -3. 4.]
		*		// Becomes
		*		[-1. 2. 3. -4.]
		*
		* \endrst
		*/
		LR_INLINE basic_ndarray<T, alloc> operator-() const
		{
			basic_ndarray<T, alloc> res(m_extent);
			arithmetic::array_op(res.m_data_start, m_data_start, m_extent,
								 res.get_stride(), m_stride,
								 [](T a)
			{
				return -a;
			});

			return res;
		}

		template<typename O>
		LR_INLINE void reshape(const basic_extent<O> &new_shape)
		{
			auto tmp_shape = new_shape.fix_automatic(m_extent_product);

			lr_int prod = math::product(tmp_shape.get_extent(), tmp_shape.ndim());
			if (prod != m_extent_product)
				throw std::length_error("Cannot reshape array with " + m_extent.str() +
										+" (" + std::to_string(m_extent_product) + " elements)" +
										" to " + new_shape.str() + " (" +
										std::to_string(prod) + " elements)");

			if (!m_stride.is_trivial())
			{
				// Non-trivial stride, so this array will be dereferenced and a new array
				// created in its place

				// This destroys the current array and replaces it!

				auto new_data = m_alloc.allocate(m_extent_product);

				lr_int idim = 0;
				lr_int dims = ndim();

				const auto *__restrict _extent = m_extent.get_extent_alt();
				const auto *__restrict _stride_this = m_stride.get_stride_alt();

				lr_int coord[LIBRAPID_MAX_DIMS]{};

				do
				{
					*(new_data++) = *m_data_start;

					for (idim = 0; idim < dims; ++idim)
					{
						if (++coord[idim] == _extent[idim])
						{
							coord[idim] = 0;
							m_data_start = m_data_start - (_extent[idim] - 1) * _stride_this[idim];
						}
						else
						{
							m_data_start = m_data_start + _stride_this[idim];
							break;
						}
					}
				} while (idim < dims);

				new_data -= m_extent_product;

				// Erase the current array
				decrement();

				// Initialize new values
				m_data_origin = new_data;
				m_data_start = new_data;

				m_origin_references = new std::atomic<lr_int>(1);

				m_origin_size = m_extent_product;
			}

			m_extent = extent(tmp_shape);
			m_stride = stride::from_extent(std::vector<O>(tmp_shape.begin(),
										   tmp_shape.end()));
			m_stride.set_contiguous(m_stride.check_contiguous(m_extent.get_extent(),
									m_extent.ndim()));
		}

		template<typename O>
		LR_INLINE void reshape(const std::initializer_list<O> &new_shape)
		{
			reshape(extent(new_shape));
		}

		template<typename O>
		LR_INLINE void reshape(const std::vector<O> &new_shape)
		{
			reshape(extent(new_shape));
		}

		template<typename O>
		LR_INLINE basic_ndarray<T, alloc> reshaped(const basic_extent<O> &new_shape) const
		{
			auto res = create_reference();
			res.reshape(new_shape);
			return res;
		}

		template<typename O>
		LR_INLINE basic_ndarray<T, alloc> reshaped(const std::initializer_list<O> &new_shape) const
		{
			return reshaped(extent(new_shape));
		}

		template<typename O>
		LR_INLINE basic_ndarray<T, alloc> reshaped(const std::vector<O> &new_shape) const
		{
			return reshaped(extent(new_shape));
		}

		LR_INLINE void strip_front()
		{
			// Remove leading dimensions which are all 1

			lr_int strip_to = 0;
			for (lr_int i = 0; i < ndim(); i++)
				if (m_extent[i] == 1) strip_to++;
				else break;

			// Ensure arrays of shape [1, 1, ... 1] are not
			// completely erased
			if (strip_to == ndim())
				strip_to--;

			lr_int new_dims = ndim() - strip_to;

			lr_int new_extent[LIBRAPID_MAX_DIMS]{};
			for (lr_int i = 0; i < new_dims; i++)
				new_extent[i] = m_extent[i + strip_to];

			lr_int new_stride[LIBRAPID_MAX_DIMS]{};
			for (lr_int i = 0; i < new_dims; i++)
				new_stride[i] = m_stride[i + strip_to];

			bool tmp = m_stride.is_contiguous();
			m_stride = stride(new_stride, new_dims);
			m_extent = extent(new_extent, new_dims);
			m_stride.set_contiguous(tmp);
		}

		LR_INLINE void strip_back()
		{
			// Remove trailing dimensions which are all 1

			lr_int strip_to = ndim();
			for (lr_int i = ndim(); i > 0; i--)
				if (m_extent[i - 1] == 1) strip_to--;
				else break;

			// Ensure arrays of shape [1, 1, ... 1] are not
			// completely erased
			if (strip_to == 0)
				strip_to++;

			lr_int new_extent[LIBRAPID_MAX_DIMS]{};
			for (lr_int i = 0; i < strip_to; i++)
				new_extent[i] = m_extent[i];

			lr_int new_stride[LIBRAPID_MAX_DIMS]{};
			for (lr_int i = 0; i < strip_to; i++)
				new_stride[i] = m_stride[i];

			bool tmp = m_stride.is_contiguous();
			m_stride = stride(new_stride, strip_to);
			m_extent = extent(new_extent, strip_to);
			m_stride.set_contiguous(tmp);
		}

		LR_INLINE void strip()
		{
			strip_front();
			strip_back();
		}

		LR_INLINE basic_ndarray<T, alloc> stripped_front() const
		{
			auto res = create_reference();
			res.strip_front();
			return res;
		}

		LR_INLINE basic_ndarray<T, alloc> stripped_back() const
		{
			auto res = create_reference();
			res.strip_back();
			return res;
		}

		LR_INLINE basic_ndarray<T, alloc> stripped() const
		{
			auto res = create_reference();
			res.strip();
			return res;
		}

		template<typename O>
		LR_INLINE void transpose(const std::vector<O> &order)
		{
			// Validate the ordering
			if (order.size() != (size_t) ndim())
			{
				std::string msg = "To transpose an array with " + std::to_string(ndim()) + " dimensions, "
					+ std::to_string(ndim()) + " indices are required, but only " +
					std::to_string(order.size()) + " were supplied";
				throw std::domain_error(msg);
			}

			bool valid = true;
			std::vector<O> missing;
			for (lr_int i = 0; i < ndim(); i++)
			{
				if (std::count(order.begin(), order.end(), (O) i) != 1)
				{
					missing.emplace_back(i);
					valid = false;
				}
			}

			if (!valid)
			{
				auto stream = std::stringstream();
				for (lr_int i = 0; i < m_stride.ndim(); i++)
				{
					if (i == m_stride.ndim() - 1) stream << m_stride.get_stride()[i];
					else stream << m_stride.get_stride()[i] << ", ";
				}
				std::string missing_str = "(" + stream.str() + ")";

				std::string msg = "Transpose requires that each index is passed exactly once, but indices "
					+ missing_str + " were passed more than once or not at all";
				throw std::runtime_error(msg);
			}

			m_extent.reshape(order);
			m_stride.reshape(order);
		}

		LR_INLINE void transpose()
		{
			std::vector<lr_int> order(ndim());
			for (lr_int i = 0; i < ndim(); i++)
				order[i] = ndim() - i - 1;
			transpose(order);
		}

		template<typename O>
		LR_INLINE void transpose(const std::initializer_list<O> &order)
		{
			transpose(std::vector<O>(order.begin(), order.end()));
		}

		template<typename O>
		LR_INLINE basic_ndarray<T, alloc> transposed(const std::vector<O> &order) const
		{
			auto res = create_reference();
			res.transpose(order);
			return res;
		}

		template<typename O>
		LR_INLINE basic_ndarray<T, alloc> transposed(const std::initializer_list<O> &order) const
		{
			return transposed(order);
		}

		LR_INLINE basic_ndarray<T, alloc> transposed() const
		{
			auto res = create_reference();
			res.transpose();
			return res;
		}

		/**
		 * \rst
		 *
		 * Calculate the sum of the values in an array and return
		 * the result.
		 *
		 * Passing in no parameters (or passing in ``axis=AUTO``)
		 * will calculate the sum of the entire array and will
		 * return a scalar value (of type basic_ndarray) with the
		 * result.
		 *
		 * Setting the ``axis`` parameter will calculate the sum
		 * over a particular axis. For example, calculating the
		 * sum of a matrix over the first axis (``axis=0``) will
		 * return a new array with the same number of columns as
		 * the original matrix, where each value is the sum of
		 * the values in the corresponding column.
		 *
		 * .. Hint::
		 *		To convert a zero-dimensional array (a scalar) to
		 *		the corresponding arithmetic type, you can use the
		 *		function ``my_array.to_scalar()``
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[[ 1.  2.  3.  4.]
		 *		 [ 5.  6.  7.  8.]
		 *		 [ 9. 10. 11. 12.]]
		 *
		 *		# The resulting vector after "sum(0)"
		 *		[15. 18. 21. 24.]
		 *
		 * The datatype of the returned array will be the same
		 * as the type of the input array.
		 *
		 * Parameters
		 * ----------
		 *
		 * axis = AUTO: any arithmetic type
		 * 		The axis to calculate the sum over
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray (potentially a scalar)
		 * 		The sum of the array (over an axis)
		 *
		 * \endrst
		 */
		template<typename A = lr_int, typename std::enable_if<std::is_integral<A>::value, int>::type = 0>
		LR_INLINE basic_ndarray<T> sum(A axis = AUTO) const
		{
			return basic_ndarray<T>::recursive_axis_func(*this, [&]<typename V>(const basic_ndarray<V> &arr)
			{
				V res = 0;
				basic_ndarray<V> fixed_array;
				V *__restrict fixed;

				if (arr.get_stride().is_trivial())
				{
					fixed = arr.get_data_start();
				}
				else
				{
					fixed_array = arr.reshaped({AUTO}).clone();
					fixed = fixed_array.get_data_start();
				}

				for (lr_int i = 0; i < arr.size(); i++)
					res += fixed[i];
				return basic_ndarray<V>::from_data(res);
			}, axis, 0);
		}

		/**
		 * \rst
		 *
		 * Calculate the product of the values in an array and
		 * return the result.
		 *
		 * Passing in no parameters (or passing in ``axis=AUTO``)
		 * will calculate the product of the entire array and will
		 * return a scalar value (of type basic_ndarray) with the
		 * result.
		 *
		 * Setting the ``axis`` parameter will calculate the
		 * product over a particular axis. For example, calculating
		 * the sum of a matrix over the first axis (``axis=0``) will
		 * return a new array with the same number of columns as
		 * the original matrix, where each value is the product of
		 * the values in the corresponding column.
		 *
		 * .. Hint::
		 *		To convert a zero-dimensional array (a scalar) to
		 *		the corresponding arithmetic type, you can use the
		 *		function ``my_array.to_scalar()``
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[[ 1.  2.  3.  4.]
		 *		 [ 5.  6.  7.  8.]
		 *		 [ 9. 10. 11. 12.]]
		 *
		 *		# The resulting vector after "product(0)"
		 *		[ 45. 120. 231. 384.]
		 *
		 * The datatype of the returned array will be the same
		 * as the type of the input array.
		 *
		 * Parameters
		 * ----------
		 *
		 * axis = AUTO: any arithmetic type
		 * 		The axis to calculate the product over
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray (potentially a scalar)
		 * 		The product of the array (over an axis)
		 *
		 * \endrst
		 */
		template<typename A = lr_int, typename std::enable_if<std::is_integral<A>::value, int>::type = 0>
		LR_INLINE basic_ndarray<T> product(A axis = AUTO) const
		{
			return basic_ndarray<T>::recursive_axis_func(*this, [&]<typename V>(const basic_ndarray<V> &arr)
			{
				V res = 1;
				basic_ndarray<V> fixed_array;
				V *__restrict fixed;

				if (arr.get_stride().is_trivial())
				{
					fixed = arr.get_data_start();
				}
				else
				{
					fixed_array = arr.reshaped({AUTO}).clone();
					fixed = fixed_array.get_data_start();
				}

				for (lr_int i = 0; i < arr.size(); i++)
					res *= fixed[i];
				return basic_ndarray<V>::from_data(res);
			}, axis, 0);
		}

		/**
		 * \rst
		 *
		 * Calculate the mean average of the values in an array and
		 * return the result.
		 *
		 * Passing in no parameters (or passing in ``axis=AUTO``)
		 * will calculate the mean of the entire array and will
		 * return a scalar value (of type basic_ndarray) with the
		 * result.
		 *
		 * Setting the ``axis`` parameter will calculate the
		 * mean over a particular axis. For example, calculating
		 * the mean of a matrix over the first axis (``axis=0``) will
		 * return a new array with the same number of columns as
		 * the original matrix, where each value is the mean average of
		 * the values in the corresponding column.
		 *
		 * .. Hint::
		 *		To convert a zero-dimensional array (a scalar) to
		 *		the corresponding arithmetic type, you can use the
		 *		function ``my_array.to_scalar()``
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[[ 1.  2.  3.  4.]
		 *		 [ 5.  6.  7.  8.]
		 *		 [ 9. 10. 11. 12.]]
		 *
		 *		# The resulting vector after "mean(0)"
		 *		[5. 6. 7. 8.]
		 *
		 * The datatype of the returned array will be the same
		 * as the type of the input array.
		 *
		 * Parameters
		 * ----------
		 *
		 * axis = AUTO: any arithmetic type
		 * 		The axis to calculate the mean over
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray (potentially a scalar)
		 * 		The mean average of the array (over an axis)
		 *
		 * \endrst
		 */
		template<typename A = lr_int, typename std::enable_if<std::is_integral<A>::value, int>::type = 0>
		LR_INLINE basic_ndarray<T> mean(A axis = AUTO) const
		{
			return basic_ndarray<T>::recursive_axis_func(*this, [&]<typename V>(const basic_ndarray<V> &arr)
			{
				V res = 0;
				basic_ndarray<V> fixed_array;
				V *__restrict fixed;

				if (arr.get_stride().is_trivial())
				{
					fixed = arr.get_data_start();
				}
				else
				{
					fixed_array = arr.reshaped({AUTO}).clone();
					fixed = fixed_array.get_data_start();
				}

				for (lr_int i = 0; i < arr.size(); i++)
					res += fixed[i];

				return basic_ndarray<V>::from_data(res / (V) arr.size());
			}, axis, 0);
		}

		/**
		* \rst
		*
		* Calculate the absolute value of each element of an array
		* and return a new array containing those values.
		*
		* .. Math::
		*
		*		y=\mid x \mid
		*
		* Example:
		*
		* .. code-block:: python
		*
		*		# The input matrix
		*		[-1 -2 -3 -4 -5]
		*
		*		# The resulting vector after "abs()"
		*		[1 2 3 4 5]
		*
		* The datatype of the returned array will be the same
		* as the type of the input array.
		*
		* Parameters
		* ----------
		*
		* None
		*
		* Returns
		* -------
		*
		* result: basic_ndarray (potentially a scalar)
		* 		An array containing the absolute values of the
		*		elements in the parent array
		*
		* \endrst
		*/
		LR_INLINE basic_ndarray<T> abs() const
		{
			return mapped([](T x)
			{
				return std::abs(x);
			});
		}

		/**
		* \rst
		*
		* Calculate the square of each element of an array
		* and return a new array containing those values.
		*
		* .. Math::
		*
		*		y=x^2
		*
		* Example:
		*
		* .. code-block:: python
		*
		*		# The input matrix
		*		[1. 2. 3. 4. 5.]
		*
		*		# The resulting vector after "abs()"
		*		[ 1.  4.  9. 16. 25.]
		*
		* The datatype of the returned array will be the same
		* as the type of the input array.
		*
		* Parameters
		* ----------
		*
		* None
		*
		* Returns
		* -------
		*
		* result: basic_ndarray (potentially a scalar)
		* 		An array containing the squared values of the
		*		elements in the parent array
		*
		* \endrst
		*/
		LR_INLINE basic_ndarray<T> square() const
		{
			return mapped([](T x)
			{
				return x * x;
			});
		}

		LR_INLINE basic_ndarray<T> sqrt() const
		{
			return mapped([](T x)
			{
				return std::sqrt(x);
			});
		}

		/**
		 * \rst
		 *
		 * Calculate the variance of the values in an array and
		 * return the result.
		 *
		 * Passing in no parameters (or passing in ``axis=AUTO``)
		 * will calculate the variance of the entire array and will
		 * return a scalar value (of type basic_ndarray) with the
		 * result.
		 *
		 * Setting the ``axis`` parameter will calculate the
		 * mean over a particular axis. For example, calculating
		 * the variance of a matrix over the first axis (``axis=0``) will
		 * return a new array with the same number of columns as
		 * the original matrix, where each value is the variance of
		 * the values in the corresponding column.
		 *
		 * The variance of an array is calculated as follows:
		 *
		 * .. Math::
		 *
		 *		\textit{mean}(x - \textit{mean}(x)) ^ 2)
		 *
		 * .. Hint::
		 *		To convert a zero-dimensional array (a scalar) to
		 *		the corresponding arithmetic type, you can use the
		 *		function ``my_array.to_scalar()``
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[[ 1.  2.  3.  4.]
		 *		 [ 5.  6.  7.  8.]
		 *		 [ 9. 10. 11. 12.]]
		 *
		 *		# The resulting vector after "variance(0)"
		 *		[10.6667 10.6667 10.6667 10.6667]
		 *
		 * The datatype of the returned array will be the same
		 * as the type of the input array.
		 *
		 * Parameters
		 * ----------
		 *
		 * axis = AUTO: any arithmetic type
		 * 		The axis to calculate the variance over
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray (potentially a scalar)
		 * 		The variance of the array (over an axis)
		 *
		 * \endrst
		 */
		template<typename A = lr_int, typename std::enable_if<std::is_integral<A>::value, int>::type = 0>
		LR_INLINE basic_ndarray<T> variance(A axis = AUTO) const
		{
			return basic_ndarray<T>::recursive_axis_func(*this, [&]<typename V>(const basic_ndarray<V> &arr)
			{
				return (arr - arr.mean()).abs().square().mean();
			}, axis, 0);
		}

		/**
		 * \rst
		 *
		 * Compare this array with another array (potentially a scalar)
		 * and return a new array with the element-wise minimum values.
		 *
		 * If both inputs are scalars, the result is a scalar.
		 *
		 * If one input is a scalar and the other is an array, the result
		 * is the minima of each element of the array and the scalar
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[1. 2. 3. 4. 5.]
		 *
		 *		# The resulting vector after "minimum(3)"
		 *		[1. 2. 3. 3. 3.]
		 *
		 * The datatype of the returned array will be the datatype
		 * with the highest precision out of the two input arrays. For
		 * example, using a datatype of ``int`` and ``float`` will
		 * return an array of ``float``s.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: basic_ndarray or scalar
		 * 		The array (or scalar) to find the element-wise minimum
		 *		with.
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray
		 * 		The element-wise minima of the input values
		 *
		 * \endrst
		 */
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			minimum(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 < x2 ? x1 : x2;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			minimum(M other) const
		{
			using ct = typename std::common_type<T, M>::type;

			return minimum(basic_ndarray<ct>::from_data((ct) other));
		}

		/**
		 * \rst
		 *
		 * Compare this array with another array (potentially a scalar)
		 * and return a new array with the element-wise maxima values.
		 *
		 * If both inputs are scalars, the result is a scalar.
		 *
		 * If one input is a scalar and the other is an array, the result
		 * is the maxima of each element of the array and the scalar
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[1. 2. 3. 4. 5.]
		 *
		 *		# The resulting vector after "maximum(3)"
		 *		[3. 3. 3. 4. 5.]
		 *
		 * The datatype of the returned array will be the datatype
		 * with the highest precision out of the two input arrays. For
		 * example, using a datatype of ``int`` and ``float`` will
		 * return an array of ``float``s.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: basic_ndarray or scalar
		 * 		The array (or scalar) to find the element-wise maximum
		 *		with.
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray
		 * 		The element-wise maxima of the input values
		 *
		 * \endrst
		 */
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			maximum(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 > x2 ? x1 : x2;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			maximum(M other) const
		{
			using ct = typename std::common_type<T, M>::type;

			return maximum(basic_ndarray<ct>::from_data((ct) other));
		}

		/**
		 * \rst
		 *
		 * Return the element-wise floor of the array
		 * Please see ``librapid::floor`` for more information
		 *
		 * \endrst
		 */
		LR_INLINE basic_ndarray<T> floor() const
		{
			auto res = clone();
			res.map([](T x)
			{
				return std::floor(x);
			});
			return res;
		}

		/**
		 * \rst
		 *
		 * Return the element-wise ceiling of the array.
		 * Please see ``librapid::ceil`` for more information
		 *
		 * \endrst
		 */
		LR_INLINE basic_ndarray<T> ceil() const
		{
			auto res = clone();
			res.map([](T x)
			{
				return std::ceil(x);
			});
			return res;
		}

		/**
		 * \rst
		 *
		 * Compare this array with another array (potentially a scalar)
		 * and return a new array containing ones and zeros only, where
		 * a value of one means the value in the left hand side array
		 * was less than the corresponding value in the right hand side
		 * array (or simply the right hand side value, if it is a scalar)
		 *
		 * If both inputs are scalars, the result is a scalar.
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[1. 2. 3. 4. 5.]
		 *
		 *		# The resulting vector after "less_than(3)"
		 *		[1. 1. 0. 0. 0.]
		 *
		 * The datatype of the returned array will be the datatype
		 * with the highest precision out of the two input arrays. For
		 * example, using a datatype of ``int`` and ``float`` will
		 * return an array of ``float``s.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: basic_ndarray or scalar
		 * 		The array (or scalar) to compare with
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray
		 * 		The element-wise result of the less than operation
		 *
		 * \endrst
		 */
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			less_than(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 < x2 ? 1 : 0;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			less_than(M other) const
		{
			using ct = typename std::common_type<T, M>::type;
			return less_than(basic_ndarray<ct>::from_data((ct) other));
		}

		/**
		 * \rst
		 *
		 * Compare this array with another array (potentially a scalar)
		 * and return a new array containing ones and zeros only, where
		 * a value of one means the value in the left hand side array
		 * was greater than the corresponding value in the right hand side
		 * array (or simply the right hand side value, if it is a scalar)
		 *
		 * If both inputs are scalars, the result is a scalar.
		 *
		 * Example:
		 *
		 * .. code-block:: python
		 *
		 *		# The input matrix
		 *		[1. 2. 3. 4. 5.]
		 *
		 *		# The resulting vector after "greater_than(3)"
		 *		[0. 0. 0. 1. 1.]
		 *
		 * The datatype of the returned array will be the datatype
		 * with the highest precision out of the two input arrays. For
		 * example, using a datatype of ``int`` and ``float`` will
		 * return an array of ``float``s.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: basic_ndarray or scalar
		 * 		The array (or scalar) to compare with
		 *
		 * Returns
		 * -------
		 *
		 * result: basic_ndarray
		 * 		The element-wise result of the greater than operation
		 *
		 * \endrst
		 */
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			greater_than(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 > x2 ? 1 : 0;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			greater_than(M other) const
		{
			using ct = typename std::common_type<T, M>::type;
			return greater_than(basic_ndarray<ct>::from_data((ct) other));
		}

		/**
		* \rst
		*
		* Compare this array with another array (potentially a scalar)
		* and return a new array containing ones and zeros only, where
		* a value of one means the value in the left hand side array
		* was less than or equal to the corresponding value in the right
		* hand side array (or simply the right hand side value, if it
		* is a scalar)
		*
		* If both inputs are scalars, the result is a scalar.
		*
		* Example:
		*
		* .. code-block:: python
		*
		*		# The input matrix
		*		[1. 2. 3. 4. 5.]
		*
		*		# The resulting vector after "less_than_or_equal(3)"
		*		[1. 1. 1. 0. 0.]
		*
		* The datatype of the returned array will be the datatype
		* with the highest precision out of the two input arrays. For
		* example, using a datatype of ``int`` and ``float`` will
		* return an array of ``float``s.
		*
		* Parameters
		* ----------
		*
		* other: basic_ndarray or scalar
		* 		The array (or scalar) to compare with
		*
		* Returns
		* -------
		*
		* result: basic_ndarray
		* 		The element-wise result of the less than or equal
		*		to operation
		*
		* \endrst
		*/
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			less_than_or_equal(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 <= x2 ? 1 : 0;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			less_than_or_equal(M other) const
		{
			using ct = typename std::common_type<T, M>::type;
			return less_than_or_equal(basic_ndarray<ct>::from_data((ct) other));
		}

		/**
		* \rst
		*
		* Compare this array with another array (potentially a scalar)
		* and return a new array containing ones and zeros only, where
		* a value of one means the value in the left hand side array
		* was greater than or equal to the corresponding value in the
		* right hand side array (or simply the right hand side value,
		* if it is a scalar)
		*
		* If both inputs are scalars, the result is a scalar.
		*
		* Example:
		*
		* .. code-block:: python
		*
		*		# The input matrix
		*		[1. 2. 3. 4. 5.]
		*
		*		# The resulting vector after "greater_than_or_equal(3)"
		*		[0. 0. 1. 1. 1.]
		*
		* The datatype of the returned array will be the datatype
		* with the highest precision out of the two input arrays. For
		* example, using a datatype of ``int`` and ``float`` will
		* return an array of ``float``s.
		*
		* Parameters
		* ----------
		*
		* other: basic_ndarray or scalar
		* 		The array (or scalar) to compare with
		*
		* Returns
		* -------
		*
		* result: basic_ndarray
		* 		The element-wise result of the greater than or
		*		equal to operation
		*
		* \endrst
		*/
		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			greater_than_or_equal(const basic_ndarray<M> &other) const
		{
			return basic_ndarray<T>::array_or_scalar_func(*this, other,
														  []<typename X1, typename X2> (X1 x1, X2 x2)
			{
				return x1 > x2 ? 1 : 0;
			});
		}

		template<typename M>
		LR_INLINE basic_ndarray<typename std::common_type<T, M>::type>
			greater_than_or_equal(M other) const
		{
			using ct = typename std::common_type<T, M>::type;
			return greater_than_or_equal(basic_ndarray<ct>::from_data((ct) other));
		}

		template<typename B_T, typename B_A>
		LR_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
			nd_allocator<typename std::common_type<T, B_T>::type>>
			dot(const basic_ndarray<B_T, B_A> &other) const
		{
			using R_T = typename std::common_type<T, B_T>::type;

			bool is_matrix_vector = false;

			const auto &o_e = other.get_extent();
			bool is_matrix_vector_like = utils::check_ptr_match(o_e.get_extent(), o_e.ndim(),
																utils::sub_vector(m_extent.get_extent(),
																m_extent.ndim(), 1), true);

			// Check for column-vector
			if (ndim() == 2 && other.ndim() == 1)
				if (m_extent[1] == o_e[0])
					is_matrix_vector = true;

			// Check for column-vector
			if (ndim() == 2 && other.ndim() == 2)
				if (m_extent[1] == o_e[0] && o_e[1] == 1)
					is_matrix_vector = true;

			if (is_matrix_vector || is_matrix_vector_like)
			{
				// Matrix-Vector product
				lr_int res_shape[LIBRAPID_MAX_DIMS]{};
				res_shape[0] = m_extent[0];

				lr_int dims = other.ndim();
				for (lr_int i = 1; i < dims; i++)
					res_shape[i] = other.get_extent()[i];

				auto res = basic_ndarray<R_T>(extent(res_shape, dims));

				// #ifdef LIBRAPID_CBLAS
				const auto M = m_extent[0];
				const auto N = m_extent[1];

				if (!is_matrix_vector && is_matrix_vector_like)
				{
					for (lr_int i = 0; i < M; i++)
						res[i] = subscript(i).dot(other);
					return res;
				}

				if (!m_stride.is_contiguous())
					return clone().dot(other);

				const R_T alpha = 1.0;
				const R_T beta = 0.0;

				const auto trans = !m_stride.is_trivial();

				const auto lda = N; // m_stride[0];
				const auto ldb = other.get_stride()[other.ndim() - 1];

				auto *__restrict a = m_data_start;
				auto *__restrict b = other.get_data_start();
				auto *__restrict c = res.get_data_start();

				if (!trans)
					linalg::cblas_gemv('r', trans, M, N, alpha, a, lda, b, ldb, beta, c, 1);
				else
					linalg::cblas_gemv_no_blas('r', trans, M, N, alpha, a, lda, b, ldb, beta, c, 1);

				return res;
			}

			if (ndim() != other.ndim())
				throw std::domain_error("Cannot compute dot product on arrays with " +
										m_extent.str() + " and " + other.get_extent().str());

			lr_int dims = ndim();

			switch (dims)
			{
				case 1:
					{
						if (m_extent[0] != other.get_extent()[0])
							throw std::domain_error("Cannot compute dot product with arrays with " +
													m_extent.str() + " and " + other.get_extent().str());

						// Vector product
						basic_ndarray<R_T, nd_allocator<R_T>> res(extent({1}));
						res.m_is_scalar = true;

						*res.get_data_start() = linalg::cblas_dot(m_extent_product, m_data_start, m_stride[0],
																  other.get_data_start(), other.get_stride()[0]);

						return res;
					}
				case 2:
					{
						if (m_extent[1] != other.get_extent()[0])
							throw std::domain_error("Cannot compute dot product with arrays with " +
													m_extent.str() + " and " + other.get_extent().str());

						// If the data is not contiguous in memory, we cannot operate on it
						// normally, so clone the arrays to reorder the data and make it
						// optimal in memory
						if (!m_stride.is_contiguous() && other.get_stride().is_contiguous())
							return clone().dot(other);

						if (m_stride.is_contiguous() && !other.get_stride().is_contiguous())
							return dot(other.clone());

						const auto M = m_extent[0];           // Rows of op(a)
						const auto N = other.get_extent()[1]; // Cols of op(b)
						const auto K = m_extent[1]; // Cols of op(a) and rows of op(b)

						const R_T alpha = 1.0;
						const R_T beta = 0.0;

						auto res = basic_ndarray<R_T>(extent{M, N});

						const auto transA = !m_stride.is_trivial();
						const auto transB = !other.get_stride().is_trivial();

						const auto lda = transA ? M : K;
						const auto ldb = transB ? K : N;
						const auto ldc = transB ? M : N;

						auto *__restrict a = m_data_start;
						auto *__restrict b = other.get_data_start();
						auto *__restrict c = res.get_data_start();

						linalg::cblas_gemm('r', transA, transB, M, N, K, alpha, a, lda,
										   b, ldb, beta, c, ldc);

						return res;
					}
				default:
					{
						// Check the arrays are valid
						if (m_extent[ndim() - 1] != other.get_extent()[other.ndim() - 2])
							throw std::domain_error("Cannot compute dot product with arrays with " +
													m_extent.str() + " and " + other.get_extent().str());

						// Create the new array dimensions
						lr_int new_extent[LIBRAPID_MAX_DIMS]{};

						// Initialize the new dimensions
						for (lr_int i = 0; i < ndim() - 1; i++) new_extent[i] = m_extent[i];
						for (lr_int i = 0; i < other.ndim() - 2; i++) new_extent[i + ndim()] = other.get_extent()[i];
						new_extent[ndim() + other.ndim() - 4] = other.get_extent()[other.ndim() - 3];
						new_extent[ndim() + other.ndim() - 3] = other.get_extent()[other.ndim() - 1];

						auto res = basic_ndarray<R_T, nd_allocator<R_T>>(extent(new_extent, ndim() + other.ndim() - 2));
						R_T *__restrict res_ptr = res.get_data_start();

						lr_int idim = 0;
						lr_int dims = res.ndim();

						const auto *__restrict _extent = res.get_extent().get_extent();
						const auto *__restrict _stride = res.get_stride().get_stride();

						lr_int coord[LIBRAPID_MAX_DIMS]{};

						std::vector<lr_int> lhs_index(ndim());
						std::vector<lr_int> rhs_index(other.ndim());

						do
						{
							// Extract the index for the lhs
							for (lr_int i = 0; i < ndim() - 1; i++)
								lhs_index[i] = coord[i];
							lhs_index[ndim() - 1] = AUTO;

							// Extract the index for the rhs
							for (lr_int i = 0; i < other.ndim() - 2; i++)
								rhs_index[i] = coord[ndim() + i - 1];
							rhs_index[other.ndim() - 2] = AUTO;
							rhs_index[other.ndim() - 1] = coord[dims - 1];

							*res_ptr = *(subarray(lhs_index).dot(other.subarray(rhs_index)).get_data_start());

							for (idim = 0; idim < dims; ++idim)
							{
								if (++coord[idim] == _extent[idim])
								{
									res_ptr = res_ptr - (_extent[idim] - 1) * _stride[idim];
									coord[idim] = 0;
								}
								else
								{
									res_ptr = res_ptr + _stride[idim];
									break;
								}
							}
						} while (idim < dims);

						res_ptr -= math::product(res.get_extent().get_extent(), res.ndim());

						return res;
					}
			}
		}

		LR_INLINE std::string str(lr_int indent = 0, bool show_commas = false) const
		{
			std::pair<lr_int, lr_int> longest;
			return stringify(indent, show_commas, false, true, longest);
		}

	private:
		LR_INLINE errors construct_new()
		{
			if (ndim() > LIBRAPID_MAX_DIMS)
				return errors::ARRAY_DIMENSIONS_TOO_LARGE;

			m_data_start = m_alloc.allocate(m_extent_product);
			m_origin_size = m_extent_product;
			m_data_origin = m_data_start;

			m_origin_references = new std::atomic<lr_int>(1);

			return errors::ALL_OK;
		}

		template<typename E, typename S>
		LR_INLINE errors construct_new(const basic_extent<E> &e, const basic_stride<S> &s)
		{
			m_extent = e;
			m_stride = s;
			m_stride.set_contiguous(m_stride.check_contiguous(m_extent.get_extent(),
									m_extent.ndim()));

			if (ndim() > LIBRAPID_MAX_DIMS)
				return errors::ARRAY_DIMENSIONS_TOO_LARGE;

			m_extent_product = math::product(m_extent.get_extent(), ndim());

			m_data_start = m_alloc.allocate(m_extent_product);
			m_origin_size = m_extent_product;

			m_data_origin = m_data_start;
			m_origin_references = new std::atomic<lr_int>(1);

			return errors::ALL_OK;
		}

		template<typename E, typename S>
		LR_INLINE errors construct_hollow(const basic_extent<E> &e, const basic_stride<S> &s)
		{
			m_extent = e;
			m_stride = s;
			m_stride.set_contiguous(m_stride.check_contiguous(m_extent.get_extent(),
									m_extent.ndim()));

			if (ndim() > LIBRAPID_MAX_DIMS)
				return errors::ARRAY_DIMENSIONS_TOO_LARGE;

			m_extent_product = math::product(m_extent.get_extent(), m_extent.ndim());
			m_origin_size = m_extent_product;

			return errors::ALL_OK;
		}

		LR_INLINE basic_ndarray<T, alloc> create_reference() const
		{
			basic_ndarray<T, alloc> res;

			res.m_data_origin = m_data_origin;
			res.m_origin_references = m_origin_references;
			res.m_origin_size = m_origin_size;

			res.m_data_start = m_data_start;

			res.m_stride = m_stride;
			res.m_extent = m_extent;

			res.m_extent_product = m_extent_product;
			res.m_is_scalar = m_is_scalar;

			increment();
			return res;
		}

		LR_INLINE void increment() const
		{
			if (!is_initialized())
				return;

			++(*m_origin_references);
		}

		LR_INLINE void decrement()
		{
			if (!is_initialized())
				return;

			--(*m_origin_references);

			if ((*m_origin_references) == 0)
			{
				m_alloc.deallocate(m_data_origin, m_origin_size);
				delete m_origin_references;
			}
		}

		LR_INLINE std::pair<lr_int, lr_int> stringify_decimal_finder(bool strip_middle,
																	 bool auto_strip) const
		{
			if (auto_strip)
			{
				if (m_extent_product >= 1000) strip_middle = true;

				// Edge case for row and column vectors
				if (ndim() == 1) strip_middle = false;
				else if (ndim() == 2 && m_extent[1] == 1) strip_middle = false;
				else if (ndim() == 2 && m_extent[0] == 1) strip_middle = false;
			}

			// Scalar values
			if (m_is_scalar)
			{
				std::stringstream stream;
				stream.precision(10);

				stream << *m_data_start;

				std::string str = stream.str();
				if (std::is_floating_point<T>::value && str.find_last_of('.') == std::string::npos)
					return {str.length(), 0};

				size_t index = str.find_last_of('.');
				if (index == std::string::npos)
					return {str.length(), 0};
				return {index, str.length() - index - 1};
			}

			lr_int longest_integral = 0, longest_decimal = 0;

			// Vectors
			if (ndim() == 1)
			{
				lr_int index = 0;

				for (lr_int i = 0; i < m_extent_product; ++i, ++index)
				{
					if (strip_middle && i == 3)
						i = m_extent_product - 3;

					auto sublongest = subscript(i).stringify_decimal_finder(strip_middle, false);

					if (sublongest.first > longest_integral) longest_integral = sublongest.first;
					if (sublongest.second > longest_decimal) longest_decimal = sublongest.second;
				}

				return {longest_integral, longest_decimal};
			}

			// Everything else
			lr_int index = 0;
			lr_int vec_size = strip_middle ? 6 : m_extent[0];
			std::string res = "[";

			for (lr_int i = 0; i < m_extent[0]; ++i, ++index)
			{
				if (strip_middle && i == 3)
					i = m_extent[0] - 3;

				auto sublongest = subscript(i).stringify_decimal_finder(strip_middle, false);

				if (sublongest.first > longest_integral) longest_integral = sublongest.first;
				if (sublongest.second > longest_decimal) longest_decimal = sublongest.second;
			}

			return {longest_integral, longest_decimal};
		}

		LR_INLINE std::string stringify(lr_int indent, bool show_commas, bool strip_middle, bool auto_strip,
										std::pair<lr_int, lr_int> &longest) const
		{
			// Non-initialized arrays
			if (!is_initialized())
				return "[NONE]";

			if (auto_strip)
			{
				if (m_extent_product >= 1000) strip_middle = true;

				// Edge case for row and column vectors
				if (ndim() == 1) strip_middle = false;
				else if (ndim() == 2 && m_extent[1] == 1) strip_middle = false;
				else if (ndim() == 2 && m_extent[0] == 1) strip_middle = false;
			}

			// Scalar values
			if (m_is_scalar)
			{
				std::stringstream stream;
				stream.precision(10);

				stream << *m_data_start;

				std::string str = stream.str();
				if (std::is_floating_point<T>::value && str.find_last_of('.') == std::string::npos)
					str += ".";

				return str;
			}

			// Find the numbers being printed with the most
			// digits before and after the decimal point
			if (longest.first == 0 && longest.second == 0)
				longest = stringify_decimal_finder(false, true);

			// Vectors
			if (ndim() == 1)
			{
				lr_int index = 0;
				lr_int vec_size = strip_middle ? 6 : m_extent_product;
				std::string res = "[";

				for (lr_int i = 0; i < m_extent_product; ++i, ++index)
				{
					if (strip_middle && i == 3)
					{
						i = m_extent_product - 3;
						res += "... ";
					}

					std::string temp_val = subscript(i).stringify(indent + 1, show_commas,
																  strip_middle, false, longest);

					// Locate the decimal point and calculate
					// the number of digits before and after it
					lr_int before = 0, after = 0;
					auto index = temp_val.find('.');

					if (index == std::string::npos)
					{
						// No decimal point
						before = temp_val.length();
						after = 0;
					}
					else
					{
						before = index;
						after = temp_val.length() - index - 1;
					}

					lr_int to_add_before, to_add_after;
					to_add_before = longest.first - before;
					to_add_after = longest.second - after;

					res += std::string(to_add_before, ' ');
					res += temp_val;
					res += std::string(to_add_after, ' ');

					if (i + 1 < m_extent_product) res += show_commas ? ", " : " ";
				}
				return res + "]";
			}

			// Everything else
			lr_int index = 0;
			lr_int vec_size = strip_middle ? 6 : m_extent[0];
			std::string res = "[";

			for (lr_int i = 0; i < m_extent[0]; ++i, ++index)
			{
				if (strip_middle && i == 3)
				{
					i = m_extent[0] - 3;
					res += "...\n" + std::string(indent + 1, ' ');
				}

				res += subscript(i).stringify(indent + 1, show_commas,
											  strip_middle, false, longest);
				if (i + 1 < m_extent[0])
				{
					res += std::string((ndim() > 2) + 1, '\n');
					res += std::string(indent + 1, ' ');
				}
			}
			return res + "]";
		}

		LR_INLINE const basic_ndarray<T, alloc> subscript(lr_int index) const
		{
			if (index < 0 || index >= m_extent.get_extent()[0])
			{
				std::string msg = "Index " + std::to_string(index) +
					" out of range for array with leading dimension "
					+ std::to_string(m_extent.get_extent()[0]);

				throw std::out_of_range(msg);
			}

			basic_ndarray<T, alloc> res;
			res.m_data_origin = m_data_origin;

			res.m_data_start = m_data_start + m_stride.get_stride()[0] * index;
			res.m_origin_references = m_origin_references;
			lr_int dims = ndim();

			lr_int new_extent[LIBRAPID_MAX_DIMS]{};
			lr_int new_stride[LIBRAPID_MAX_DIMS]{};

			if (dims == 1)
			{
				// Return a scalar value
				new_extent[0] = 1;
				new_stride[0] = 1;
				res.construct_hollow(extent(new_extent, 1), stride(new_stride, 1));
				res.m_is_scalar = true;
			}
			else
			{
				memcpy(new_extent, m_extent.get_extent() + 1, sizeof(lr_int) * (LIBRAPID_MAX_DIMS - 1));
				memcpy(new_stride, m_stride.get_stride() + 1, sizeof(lr_int) * (LIBRAPID_MAX_DIMS - 1));

				res.construct_hollow(extent(new_extent, m_extent.ndim() - 1),
									 stride(new_stride, m_stride.ndim() - 1));

				res.m_is_scalar = false;
			}

			increment();
			return res;
		}

		LR_INLINE basic_ndarray<T, alloc> transposed_matrix() const
		{
			if (ndim() != 2)
				throw std::domain_error("Cannot matrix transpose array with shape " + m_extent.str());

			basic_ndarray<T, alloc> res(extent{m_extent[1], m_extent[0]});
			lr_int lda = m_stride[0], fda = m_stride[1];
			lr_int scal = m_extent[1];

			for (lr_int i = 0; i < m_extent[0]; i++)
				for (lr_int j = 0; j < m_extent[1]; j++)
					res.set_value(i + j * scal, m_data_start[j * fda + i * lda]);

			return res;
		}

		template<typename A_T, typename A_A, typename B_T, typename B_A, typename LAMBDA>
		static LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			array_array_arithmetic(const basic_ndarray<A_T, A_A> &a,
								   const basic_ndarray<B_T, B_A> &b, LAMBDA op)
		{
			using C = typename std::common_type<A_T, B_T>::type;
			using R = nd_allocator<typename std::common_type<A_T, B_T>::type>;

			auto mode = broadcast::calculate_arithmetic_mode(a.get_extent().get_extent(),
															 a.ndim(), b.get_extent().get_extent(), b.ndim());

			if (mode == (lr_int) -1)
			{
				auto msg = std::string("Cannot operate arrays with shapes ")
					+ a.get_extent().str() + " and " + b.get_extent().str();
				throw std::length_error(msg);
			}

			switch (mode)
			{
				case 0:
					{
						// Cases:
						//  > Exact match
						//  > End dimensions of other match this
						//  > End dimensions of this match other

						auto tmp_a = a.stripped();
						auto tmp_b = b.stripped();

						basic_ndarray<C, R> res(a.get_extent());
						arithmetic::array_op_array(tmp_a.get_data_start(), tmp_b.get_data_start(), res.get_data_start(),
												   tmp_a.get_extent(),
												   tmp_a.get_stride(), tmp_b.get_stride(), res.get_stride(),
												   op);
						return res;
					}
				case 1:
					{
						// Cases:
						//  > Other is a single value

						basic_ndarray<C, R> res(a.get_extent());
						arithmetic::array_op_scalar(a.get_data_start(), b.get_data_start(), res.get_data_start(),
													a.get_extent(),
													a.get_stride(), res.get_stride(),
													op);
						return res;
					}
				case 2:
					{
						// Cases:
						//  > This is a single value

						basic_ndarray<C, R> res(b.get_extent());
						arithmetic::scalar_op_array(a.get_data_start(), b.get_data_start(), res.get_data_start(),
													b.get_extent(),
													b.get_stride(), res.get_stride(),
													op);
						return res;
					}
				case 3:
					{
						// Cases:
						//  > "Row by row" operations

						auto res = basic_ndarray<C, R>(a.get_extent());

						for (lr_int i = 0; i < a.get_extent()[0]; i++)
							res[i] = op(a[i], b);

						return res;
					}
				case 4:
					{
						// Cases:
						//  > Reverse "row by row" operations

						auto res = basic_ndarray<C, R>(b.get_extent());

						for (lr_int i = 0; i < b.get_extent()[0]; i++)
							res[i] = op(a, b[i]);

						return res;
					}
				case 5:
					{
						// Cases
						//  > Grid operations

						extent res_shape(b.ndim() + 1);
						for (lr_int i = 0; i < b.ndim(); i++)
							res_shape[i] = a.get_extent()[i];
						res_shape[b.ndim()] = b.get_extent()[b.ndim() - 1];

						auto res = basic_ndarray<C, R>(res_shape);

						for (lr_int i = 0; i < res_shape[0]; i++)
							res[i] = op(a[i], b);

						return res;
					}
				case 6:
					{
						// Cases
						//  > Reverse grid operations

						extent res_shape(a.ndim() + 1);
						for (lr_int i = 0; i < a.ndim(); i++)
							res_shape[i] = b.get_extent()[i];
						res_shape[a.ndim()] = a.get_extent()[a.ndim() - 1];

						auto res = basic_ndarray<C, R>(res_shape, a.ndim() + 1);

						for (lr_int i = 0; i < res_shape[0]; i++)
							res[i] = op(a, b[i]);

						return res;
					}
				case 7:
					{
						// Cases
						//  > "Column by column" operations

						if (b.ndim() == 2)
							return op(a.transposed(), b.transposed().stripped()).transposed();

						lr_int new_extent[LIBRAPID_MAX_DIMS]{};
						for (lr_int i = 0; i < a.ndim(); i++)
							new_extent[i] = a.get_extent()[i];

						auto res = basic_ndarray<C, R>(extent(new_extent, a.ndim()));

						for (lr_int i = 0; i < new_extent[0]; i++)
							res[i] = op(a[i], b[i]);

						return res;
					}
				case 8:
					{
						// Cases:
						// Check for reverse "column by column" operations

						if (a.ndim() == 2)
							return op(a.transposed().stripped(), b.transposed()).transposed();

						lr_int new_extent[LIBRAPID_MAX_DIMS]{};
						for (lr_int i = 0; i < b.ndim(); i++)
							new_extent[i] = b.get_extent()[i];

						auto res = basic_ndarray<C, R>(extent(new_extent, b.ndim()));

						for (lr_int i = 0; i < new_extent[0]; i++)
							res[i] = op(a[i], b[i]);

						return res;
					}
				default:
					{
						auto msg = std::string("Arithmetic mode ") + std::to_string(mode) +
							" is not yet implemented, so cannot operate on arrays of shape " + a.get_extent().str() +
							" and " + b.get_extent().str();
						throw std::runtime_error(msg);
					}
			}
		}

		template<typename A_T, typename A_A, typename B_T, typename LAMBDA>
		static LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			array_scalar_arithmetic(const basic_ndarray<A_T, A_A> &a, const B_T &b, LAMBDA op)
		{
			using C = typename std::common_type<A_T, B_T>::type;
			using R = nd_allocator<typename std::common_type<A_T, B_T>::type>;

			auto res = basic_ndarray<C, R>(a.get_extent());
			arithmetic::array_op_scalar(a.get_data_start(), &b, res.get_data_start(),
										a.get_extent(), a.get_stride(), res.get_stride(), op);
			return res;
		}

		template<typename A_T, typename B_T, typename B_A, typename LAMBDA>
		static LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			scalar_array_arithmetic(const A_T &a, const basic_ndarray<B_T, B_A> &b, LAMBDA op)
		{
			using C = typename std::common_type<A_T, B_T>::type;
			using R = nd_allocator<typename std::common_type<A_T, B_T>::type>;

			auto res = basic_ndarray<C, R>(b.get_extent());
			arithmetic::scalar_op_array(&a, b.get_data_start(), res.get_data_start(),
										b.get_extent(), b.get_stride(), res.get_stride(), op);
			return res;
		}

		template<typename A_T, typename A_A, typename B_T, typename B_A, typename LAMBDA>
		static LR_INLINE void array_array_arithmetic_inplace(basic_ndarray<A_T, A_A> &a,
															 const basic_ndarray<B_T, B_A> &b,
															 LAMBDA op)
		{
			auto mode = broadcast::calculate_arithmetic_mode(a.get_extent().get_extent(), a.ndim(),
															 b.get_extent().get_extent(), b.ndim());

			if (mode == (lr_int) -1)
			{
				auto msg = std::string("Cannot operate arrays with shapes ")
					+ a.get_extent().str() + " and " + b.get_extent().str();
				throw std::length_error(msg);
			}

			switch (mode)
			{
				case 0:
					{
						// Cases:
						//  > Exact match
						//  > End dimensions of other match this
						//  > End dimensions of this match other

						auto tmp_a = a.stripped();
						auto tmp_b = b.stripped();

						arithmetic::array_op_array(tmp_a.get_data_start(), tmp_b.get_data_start(),
												   a.get_data_start(), tmp_a.get_extent(),
												   tmp_a.get_stride(), tmp_b.get_stride(),
												   a.get_stride(), op);
						break;
					}
				case 1:
					{
						// Cases:
						//  > Other is a single value

						arithmetic::array_op_scalar(a.get_data_start(), b.get_data_start(),
													a.get_data_start(), a.get_extent(),
													a.get_stride(), a.get_stride(), op);
						break;
					}
				case 3:
					{
						// Cases:
						//  > "Row by row" operation

						for (lr_int i = 0; i < a.get_extent()[0]; i++)
							a[i] = op(a[i], b);
						break;
					}
				case 7:
					{
						// Cases
						//  > "Column by column" operation

						// Cases
						//  > "Column by column" operation

						if (b.ndim() == 2)
						{
							a = op(a.transposed(), b.transposed().stripped()).transposed();
							break;
						}

						lr_int new_extent[LIBRAPID_MAX_DIMS]{};
						for (lr_int i = 0; i < a.ndim(); i++)
							new_extent[i] = a.get_extent()[i];

						for (lr_int i = 0; i < new_extent[0]; i++)
							a[i] = op(a[i], b[i]);

						break;
					}
				default:
					{
						auto msg = std::string("Inplace arithmetic mode ") + std::to_string(mode) +
							" is not valid on arrays of shape " + a.get_extent().str() +
							" and " + b.get_extent().str();
						throw std::runtime_error(msg);
					}
			}
		}

		template<typename A_T, typename A_A, typename B_T, typename LAMBDA>
		static LR_INLINE void array_scalar_arithmetic_inplace(const basic_ndarray<A_T, A_A> &a,
															  const B_T &b, LAMBDA op)
		{
			arithmetic::array_op_scalar(a.get_data_start(), &b, a.get_data_start(),
										a.get_extent(), a.get_stride(), a.get_stride(), op);
		}

		template<typename A_T, typename B_T, typename B_A, typename LAMBDA>
		static LR_INLINE void scalar_array_arithmetic_inplace(const A_T &a,
															  const basic_ndarray<B_T, B_A> &b,
															  LAMBDA op)
		{
			arithmetic::scalar_op_array(&a, b.get_data_start(), b.get_data_start(),
										b.get_extent(), b.get_stride(), b.get_stride(), op);
		}

		template<typename A, typename LAMBDA>
		static LR_INLINE basic_ndarray<A> recursive_axis_func(const basic_ndarray<A> &arr,
															  LAMBDA func, lr_int axis, lr_int depth)
		{
			if (arr.is_scalar())
				return func(arr.reshaped({1ll, AUTO}));

			if (axis == (lr_int) -1 || arr.ndim() == 1)
				return func(arr);

			std::vector<lr_int> transpose_order(arr.ndim());

			if (depth == 0)
			{
				for (lr_int i = axis; i < arr.ndim() - 1; i++)
					transpose_order[i] = i < axis ? i : i + 1;
				transpose_order[transpose_order.size() - 1] = axis;
			}
			else
			{
				for (lr_int i = 0; i < arr.ndim(); i++)
					transpose_order[i] = i;
			}

			auto fixed = arr.transposed(transpose_order);

			std::vector<lr_int> res_shape(arr.ndim() - 1);
			for (lr_int i = 0; i < arr.ndim() - 1; i++)
				res_shape[i] = arr.get_extent()[transpose_order[i]];

			auto res = basic_ndarray<T>(extent(res_shape));

			for (lr_int outer = 0; outer < res_shape[0]; outer++)
				res[outer] = basic_ndarray<T>::recursive_axis_func(fixed[outer], func,
																   math::max(axis, 1) - 1,
																   depth + 1);

			return res;
		}

		template<typename A, typename B, typename LAMBDA>
		LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
			array_or_scalar_func(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2,
								 LAMBDA func) const
		{
			using ct = typename std::common_type<A, B>::type;

			// If both are scalars, the result is a scalar
			if (x1.is_scalar() && x2.is_scalar())
				return basic_ndarray<ct>::from_data((ct) math::min(*x1.get_data_start(),
													*x2.get_data_start()));

			// If this is an array and other is a scalar, the
			// result is the minima of each element of the array
			// and the scalar
			if (!x1.is_scalar() && x2.is_scalar())
			{
				return basic_ndarray<ct>::
					array_scalar_arithmetic(x1, *x2.get_data_start(),
											[&]<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return func((ct) a, (ct) b);
				});
			}

			if (x1.is_scalar() && !x2.is_scalar())
			{
				return basic_ndarray<ct>::
					scalar_array_arithmetic(*x1.get_data_start(), x2,
											[&]<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return func((ct) a, (ct) b);
				});
			}

			// Both values are arrays
			if (x1.get_extent() != x2.get_extent())
				throw std::domain_error("Cannot operate arrays with "
										+ x1.get_extent().str() + " and "
										+ x2.get_extent().str()
										+ ". Arrays must be of the same size");

			basic_ndarray<ct> res(x1.get_extent());
			arithmetic::array_op_array(x1.get_data_start(), x2.get_data_start(), res.get_data_start(),
									   x1.get_extent(),
									   x1.get_stride(), x2.get_stride(), res.get_stride(),
									   func);
			return res;
		}

	private:

		extent m_extent;
		stride m_stride;

		lr_int m_extent_product = 0;

		T *m_data_start = nullptr;
		lr_int m_origin_size = 0;
		T *m_data_origin = nullptr;

		std::atomic<lr_int> *m_origin_references = nullptr;

		bool m_is_scalar = false;

		_alloc m_alloc = alloc();
	};

	/**
	 * \rst
	 *
	 * Create and return a new array with the same shape
	 * as the provided array, but filled entirely with
	 * zeros.
	 *
	 * The datatype of the returned array will be the same
	 * as the type of the input array.
	 *
	 * Parameters
	 * ----------
	 *
	 * arr: basic_ndarray
	 * 		The array to base the size off
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray
	 * 		A new array filled with zeros
	 *
	 * \endrst
	 */
	template<typename T>
	LR_INLINE basic_ndarray<T> zeros_like(const basic_ndarray<T> &arr)
	{
		auto res = basic_ndarray<T>(arr.get_extent());
		res.fill(0);
		return res;
	}

	/**
	 * \rst
	 *
	 * Create and return a new array with the same shape
	 * as the provided array, but filled entirely with
	 * ones.
	 *
	 * The datatype of the returned array will be the same
	 * as the type of the input array.
	 *
	 * Parameters
	 * ----------
	 *
	 * arr: basic_ndarray
	 * 		The array to base the size off
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray
	 * 		A new array filled with ones
	 *
	 * \endrst
	 */
	template<typename T>
	LR_INLINE basic_ndarray<T> ones_like(const basic_ndarray<T> &arr)
	{
		auto res = basic_ndarray<T>(arr.get_extent());
		res.fill(1);
		return res;
	}

	/**
	 * \rst
	 *
	 * Create and return a new array with the same shape
	 * as the provided array, but filled entirely with
	 * random numbers within the provided range.
	 *
	 * The datatype of the returned array will be the same
	 * as the type of the input array.
	 *
	 * .. Attention::
	 *		Please note that the ``librapid::math::random``
	 *		function returns values in the range ``[min, max]``
	 *		for integer values, though in the range ``[min, max)``
	 *		for floating point values (i.e. floating point values
	 *		will never exceed the value of ``max``)
	 *
	 * Parameters
	 * ----------
	 *
	 * arr: basic_ndarray
	 * 		The array to base the size off
	 * min = 0: any arithmetic type
	 *		The minimum random value
	 * max = 1: any arithmetic type
	 *		The maximum random value
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray
	 * 		A new array filled with random values in the
	 *		specified range
	 *
	 * \endrst
	 */
	template<typename T, typename MIN = double, typename MAX = double>
	LR_INLINE basic_ndarray<T> random_like(const basic_ndarray<T> &arr, MIN min = 0, MAX max = 1)
	{
		auto res = basic_ndarray<T>(arr.get_extent());
		res.fill_random(min, max);
		return res;
	}

	/**
	* \rst
	*
	* Return the result of element-wise flooring of values
	* in the array.
	*
	* Example:
	*
	* .. code-block:: python
	*
	*		[0.5 1.  1.5]
	*		# Becomes
	*		[0. 1. 1.]
	*
	* \endrst
	*/
	template<typename T>
	LR_INLINE basic_ndarray<T> floor(const basic_ndarray<T> &arr)
	{
		auto res = arr.clone();
		res.map([](T x)
		{
			return std::floor(x);
		});
		return res;
	}

	template<typename T>
	LR_INLINE basic_ndarray<T> floor(T val)
	{
		return std::floor(val);
	}

	/**
	* \rst
	*
	* Return the result of element-wise ceiling of values
	* in the array.
	*
	* Example:
	*
	* .. code-block:: python
	*
	*		[0.5 1.  1.5]
	*		# Becomes
	*		[1. 1. 2.]
	*
	* \endrst
	*/
	template<typename T>
	LR_INLINE basic_ndarray<T> ceil(const basic_ndarray<T> &arr)
	{
		auto res = arr.clone();
		res.map([](T x)
		{
			return std::ceil(x);
		});
		return res;
	}

	template<typename T>
	LR_INLINE basic_ndarray<T> ceil(T val)
	{
		return std::ceil(val);
	}

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator+(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr,
																[]<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a + b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator-(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr,
																[]<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a - b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator*(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr,
																[]<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a * b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	LR_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator/(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr,
																[]<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a / b;
		});
	}

	/**
	 * \rst
	 *
	 * Add two values together and return the result.
	 *
	 * The input values can be any type that supports
	 * addition. In general, the return type will be
	 * the higher precision of the two input types,
	 * or an n-dimensional array if one is passed.
	 *
	 * Parameters
	 * ----------
	 *
	 * addend1: any
	 * 		The left-hand side of the addition operation
	 * addend2: any
	 * 		The right-hand side of the addition operation
	 *
	 * Returns
	 * -------
	 *
	 * sum: any
	 * 	The result of the addition calculation
	 *
	 * \endrst
	 */
	template<typename T_A, typename T_B>
	LR_INLINE auto add(const T_A &addend1, const T_B &addend2)
	{
		return addend1 + addend2;
	}

	/**
	 * \rst
	 *
	 * Subtract one value from another and return the
	 * result.
	 *
	 * The input values can be any type that supports
	 * subtraction. In general, the return type will be
	 * the higher precision of the two input types,
	 * or an n-dimensional array if one is passed.
	 *
	 * Parameters
	 * ----------
	 *
	 * minuend: any
	 *		The left-hand side of the subtraction operation
	 * subtrahend: any
	 *		The right-hand side of the subtraction operation
	 *
	 * Returns
	 * -------
	 *
	 * difference: any
	 *		The result of the subtraction calculation
	 *
	 * \endrst
	 */
	template<typename T_A, typename T_B>
	LR_INLINE auto sub(const T_A &minuend, const T_B &subtrahend)
	{
		return minuend - subtrahend;
	}

	/**
	 * \rst
	 *
	 * Multiply two values together and return the result
	 *
	 * The input values can be any type that supports
	 * multiplication. In general, the return type will be
	 * the higher precision of the two input types,
	 * or an n-dimensional array if one is passed.
	 *
	 * Parameters
	 * ----------
	 *
	 * factor1: any
	 *		The left-hand side of the multiplication operation
	 * factor2: any
	 *		The right-hand side of the multiplication operation
	 *
	 * Returns
	 * -------
	 *
	 * product: any
	 *		The result of the multiplication calculation
	 *
	 * \endrst
	 */
	template<typename T_A, typename T_B>
	LR_INLINE auto mul(const T_A &factor1, const T_B &factor2)
	{
		return factor1 * factor2;
	}

	/**
	 * \rst
	 *
	 * Divide one value by another and return the result.
	 *
	 * The input values can be any type that supports
	 * division. In general, the return type will be
	 * the higher precision of the two input types,
	 * or an n-dimensional array if one is passed.
	 *
	 * .. Attention::
	 *		This function does not provide any division-
	 *		by-zero checking, so NaNs may be produced
	 *		unknowingly
	 *
	 * Parameters
	 * ----------
	 *
	 * dividend: any
	 *		The left-hand side of the division operation
	 * divisor: any
	 *		The right-hand side of the division operation
	 *
	 * Returns
	 * -------
	 *
	 * quotient: any
	 *		The result of the division calculation
	 *
	 * \endrst
	 */
	template<typename T_A, typename T_B>
	LR_INLINE auto div(const T_A &dividend, const T_B &divisor)
	{
		return dividend / divisor;
	}

	/**
	 * \rst
	 *
	 * Calculate the exponent of all the values in an
	 * array and return a new array with those values.
	 *
	 * Example:
	 *
	 * .. code-block:: Python
	 *
	 *		[[1, 2]   =>  [[  2.71828   7.38906  20.0855 ]
	 *		 [3, 4]]       [ 54.5982  148.413   403.429  ]]
	 *
	 * The array that is returned from the function will
	 * be of the same type as the input, and will have the
	 * same extent.
	 *
	 * .. Hint::
	 *		Although the extent of the returned array is
	 *		the same as the input array, the stride may be
	 *		altered to ensure that it's data is contiguous
	 *		in memory, improving performance dramatically.
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the exponentiation calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> exp(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::exp(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the sine of all the values in an
	 * array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The sine function expects the values to be
	 *		in radians rather than degrees, so ensure
	 *		to convert if this is undesired.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[1.570 3.141 4.712]   =>  [[  1.  0. -1.]
	 *        [6.282 7.853 9.424]]       [ -0.  1.  0.]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the sine calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> sin(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::sin(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the cosine of all the values in an
	 * array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The cosine function expects the values to be
	 *		in radians rather than degrees, so ensure
	 *		to convert if this is undesired.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[1.570 3.141 4.712]   =>  [[ 0. -1.  0.]
	 *        [6.282 7.853 9.424]]       [ 1.  0. -1.]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the cosine calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> cos(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::cos(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the tangent of all the values in an
	 * array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The tangent function expects the values to be
	 *		in radians rather than degrees, so ensure
	 *		to convert if this is undesired.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[1.570 3.141 4.712]   =>  [[ NaN  0.  NaN]
	 *        [6.282 7.853 9.424]]       [ 0.  NaN   0.]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the tangent calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> tan(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::tan(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the inverse sine of all the values in an
	 * array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The inverse sine function returns values in
	 *		radians, not degrees, so ensure this is
	 *		taken into account when using this function.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[ 0.   1.   0. ]  => [[ 0.        1.5708    0.      ]
	 *		 [ 0.5  0.  -0.5]]     [ 0.523599  0.       -0.523599]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the inverse sine calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> asin(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::asin(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the inverse cosine of all the values in
	 * an array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The inverse cosine function will return values
	 *		in radians, no degrees, so ensure this is taken
	 *		into account.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[ 0.   1.   0. ]  => [[1.5708 0.     1.5708]
	 *		 [ 0.5  0.  -0.5]]     [1.0472 1.5708 2.0944]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the inverse cosine calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> acos(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::acos(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the inverse tangent of all the values in
	 * an array and return a new array with those values.
	 *
	 * .. Attention::
	 *		The inverse tangent function will return values
	 *		in radians, no degrees, so ensure this is taken
	 *		into account.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[ 0.   1.   0. ]  => [[ 0.        0.785398  0.      ]
	 *		 [ 0.5  0.  -0.5]]     [ 0.463648  0.       -0.463648]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the inverse tangent calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> atan(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::atan(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the hyperbolic sine of each element in an
	 * array, and return a new array containing those values
	 *
	 * .. Attention::
	 *		This function deals in radians, not degrees,
	 *		so ensure to convert your array before using
	 *		this function if it will lead to unwanted
	 *		results.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[-1.  -0.5  0. ]   =>  [[-1.1752   -0.521095  0.      ]
	 *		 [ 0.   0.5  1. ]]       [ 0.        0.521095  1.1752  ]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the hyperbolic sin function calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> sinh(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::sinh(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the hyperbolic cosine of each element in an
	 * array, and return a new array containing those values
	 *
	 * .. Attention::
	 *		This function deals in radians, not degrees,
	 *		so ensure to convert your array before using
	 *		this function if it will lead to unwanted
	 *		results.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[-1.  -0.5  0. ]   =>  [[1.54308 1.12763 1.     ]
	 *		 [ 0.   0.5  1. ]]       [1.      1.12763 1.54308]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the hyperbolic cosine function calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> cosh(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::cosh(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Calculate the hyperbolic tangent of each element in an
	 * array, and return a new array containing those values
	 *
	 * .. Attention::
	 *		This function deals in radians, not degrees,
	 *		so ensure to convert your array before using
	 *		this function if it will lead to unwanted
	 *		results.
	 *
	 * .. Hint::
	 *		In the future, there will be a function to
	 *		automatically convert between angle units,
	 *		so the issue above will be much easier to
	 *		avoid.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: Python
	 *
	 *		[[-1.  -0.5  0. ]   =>  [[-0.761594 -0.462117  0.      ]
	 *		 [ 0.   0.5  1. ]]       [ 0.        0.462117  0.761594]]
	 *
	 * Parameters
	 * ----------
	 *
	 * None
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the hyperbolic tangent function calculation
	 *
	 * \endrst
	 */
	template<typename A_T, typename A_A>
	LR_INLINE basic_ndarray<A_T, A_A> tanh(const basic_ndarray<A_T, A_A> &arr)
	{
		auto res = arr.clone();
		arithmetic::array_op(res.get_data_start(), arr.get_data_start(),
							 arr.get_extent(), res.get_stride(), arr.get_stride(),
							 [](A_T value)
		{
			return std::tanh(value);
		});

		return res;
	}

	/**
	 * \rst
	 *
	 * Reshape an array of values and return the result. The resulting
	 * array is linked to the parent array, so an update in one will
	 * result in an update in the other.
	 *
	 * .. Attention::
	 *		The new array must have the same number of elements as the
	 *		original array, otherwise this function will throw an error.
	 *		This means the product of all the dimensions must be equal
	 *		before and after reshaping
	 *
	 * .. Hint::
	 *		The new array doesn't need to have the same number of
	 *		dimensions as the parent array, so you could reshape
	 *		a matrix into a vector, if you wanted to.
	 *
	 * When reshaping an array, it is possible to use the
	 * ``librapid::ndarray::AUTO`` value (-1) to specify a dimension
	 * that will adjust automatically based on the other values. It
	 * will attempt to ensure that the resulting array has the same
	 * number of elements as the parent array.
	 *
	 * Example (approximate values shown):
	 *
	 * .. code-block:: c++
	 *
	 *		// Create a 2x3 matrix filled with 0s
	 *		auto my_matrix = librapid::ndarray(librapid::extent({2, 3}), 0);
	 *		auto my_vector = librapid::reshape(my_matrix, librapid::extent({6}));
	 *
	 * Parameters
	 * ----------
	 *
	 * arr: const basic_ndarray
	 *		The array to reshape
	 * new_shape: const basic_extent
	 *		The new shape for the array
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray<T, A>
	 *		The result of the hyperbolic tangent function calculation
	 *
	 * \endrst
	 */
	template<typename T, class alloc, typename O = lr_int>
	LR_INLINE basic_ndarray<T, alloc> reshape(const basic_ndarray<T, alloc> &arr,
											  const basic_extent<O> &new_shape)
	{
		return arr.reshaped(new_shape);
	}

	/**
	 * \rst
	 *
	 * Create a new vector which starts and ends with a set value, and
	 * has a set length. The values in between will increment linearly
	 * between the two starting and ending values.
	 *
	 * .. Hint::
	 *		The resulting array's datatype will be the largest of the
	 *		start and end values. For example, passing an ``int`` and
	 *		a ``float`` will result in an array of type ``float`` being
	 *		returned.
	 *
	 * .. code-block:: c++
	 *
	 *		auto my_vector = librapid::linear(1, 5, 4)
	 *		// my_vector = [1 2 3 4]
	 *
	 * Parameters
	 * ----------
	 *
	 * start: any arithmetic type
	 *		The starting value of the vector
	 * end: any arithmetic type
	 *		The ending value of the vector
	 * len: long long
	 *		The number of elements in the vector
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray
	 *		A new vector of values
	 *
	 * \endrst
	 */
	template<typename S = double, typename E = double, typename L = lr_int>
	LR_INLINE basic_ndarray<typename std::common_type<S, E>::type>
		linear(S start, E end, L len = 0)
	{
		using ct = typename std::common_type<S, E>::type;

		basic_ndarray<ct> res({len});

		ct inc = ((ct) end - (ct) start) / (ct) (len - 1);
		for (lr_int i = 0; i < len; i++)
			res[i] = (ct) start + (ct) i * inc;

		return res;
	}

	/**
	 * \rst
	 *
	 * Create a new vector of values which starts at a specified value
	 * and ends at another, going up in specific increments. The length
	 * of the array is determined by the difference between the values
	 * and the size of the increment.
	 *
	 * .. Attention::
	 *		When passing in a start value that is larger than the end
	 *		value, the increment _must_ be negative, otherwise ``[NONE]``
	 *		will be returned.
	 *
	 * .. Hint::
	 *		The resulting array's datatype will be the largest of the
	 *		start, end and increment values. For example, passing an
	 *		``int``, a ``float`` and a ``double`` will result in an
	 *		array of type ``double`` being returned.
	 *
	 * .. code-block:: c++
	 *
	 *		auto my_vector = librapid::range(1, 5, 1);
	 *		// my_vector = [1 2 3 4]
	 *
	 * Parameters
	 * ----------
	 *
	 * start: any arithmetic type
	 *		The starting value of the vector
	 * end: any arithmetic type
	 *		The ending value of the vector
	 * inc: any arithmetic type
	 *		The difference between two consecutive elements in the vector
	 *
	 * Returns
	 * -------
	 *
	 * result: basic_ndarray
	 *		A new vector of values
	 *
	 * \endrst
	 */
	template<typename S = double, typename E = double, typename I = double>
	LR_INLINE basic_ndarray<typename std::common_type<S, E, I>::type> range(S start, E end, I inc = 1)
	{
		using ct = typename std::common_type<S, E, I>::type;

		lr_int len;

		if (inc > 0 && start < end)
			len = (lr_int) std::ceil(abs((ct) end - (ct) start) / (ct) inc);
		else if (inc > 0 && start >= end)
			len = 0;
		else if (inc < 0 && start > end)
			len = (lr_int) std::ceil(abs((ct) start - (ct) end) / (ct) -inc);
		else
			len = 0;

		auto res = basic_ndarray<ct>({len});

		for (lr_int i = 0; i < len; i++)
			res[i] = (ct) start + (ct) inc * (ct) i;

		return res;
	}

	/**
	 * \rst
	 *
	 * Given a list of arrays, join them together along an existing axis, and return the
	 * result.
	 *
	 * Parameters
	 * ----------
	 *
	 * arrays: vector, list, tuple
	 *		The list of arrays to concatenate
	 * axis = 0: integer
	 *		The axis along which to concatenate the arrays
	 *
	 * Returns
	 * -------
	 *
	 * stacked: ndarray
	 *		The stacked array
	 *
	 * .. Attention::
	 *		The arrays must have exactly the same extent, other than the dimension for the
	 *		concatenating axis, which can be different for each array.
	 *
	 * Examples
	 * --------
	 *
	 * .. code-block:: python
	 *
	 *		# Create the example arrays
	 *		first = librapid.from_data(
	 *			[[1, 2, 3],
	 *			 [4, 5, 6]]
	 *		)
	 *
	 *		second = librapid.from_data(
	 *			[[7, 8, 9]]
	 *		)
	 *
	 *		# Concatenate the arrays and store the result
	 *		# (default axis is 0 -- stack along rows)
	 *		concatenated = librapid.concatenate((first, second))
	 *
	 *		print(concatenated)
	 *
	 *		"""
	 *		Gives:
	 *
	 *		[[1. 2. 3.]
	 * 		 [4. 5. 6.]
	 * 		 [7. 8. 9.]]
	 *		"""
	 *
	 * .. code-block:: python
	 *
	 *		# Create the example arrays
	 *		first = librapid.from_data(
	 *			[[1, 2, 3],
	 *			 [4, 5, 6],
	 *			 [7, 8, 9]]
	 *		)
	 *
	 *		second = librapid.ndarray(librapid.extent(3, 2)).filled_random(-1, 1)
	 *
	 *		# Concatenate the arrays and store the result
	 *		# (here, we are stacking on axis=1 -- along the columns)
	 *		concatenated = librapid.concatenate((first, second), 1)
	 *
	 *		print(concatenated)
	 *
	 *		"""
	 *		Could give:
	 *
	 *		[[ 1.             2.             3.             0.06195223331 -0.1879928112 ]
	 *		 [ 4.             5.             6.             0.3225619793  -0.1187359691 ]
	 *		 [ 7.             8.             9.             0.2637588978  -0.1868984699 ]]
	 *		"""
	 *
	 * .. Hint::
	 *		The resulting array is contiguous in memory, meaning the resulting array
	 *		can be operated on very quickly and efficiently. (see ``ndarray.clone`` for
	 *		more information)
	 *
	 * \endrst
	 */
	template<typename A>
	LR_INLINE basic_ndarray<A> concatenate(const std::vector<basic_ndarray<A>> &arrays, lr_int axis = 0)
	{
		if (arrays.size() == 0) return {};
		if (arrays.size() == 1) return arrays[0].clone();

		// Check the arrays are all the correct size (and calculate the new dimension
		// for the result array)

		lr_int dims = arrays[0].ndim();
		lr_int new_dim = axis < dims ? arrays[0].get_extent()[axis] : 0;

		extent dims_0 = arrays[0].get_extent();

		for (lr_int i = 1; i < arrays.size(); ++i)
		{
			if (arrays[i].ndim() != dims)
				throw std::domain_error("All arrays must have the same dimensions in order to"
										" concatenate. Array at index 0 has " + std::to_string(dims) +
										", while array at index " + std::to_string(i) + " has " +
										std::to_string(arrays[i].ndim()) + " dimensions.");

			extent dims_n = arrays[i].get_extent();

			for (lr_int j = 0; j < dims; ++j)
			{
				if (j != axis && dims_0[j] != dims_n[j])
				{
					throw std::domain_error("All array dimensions other than the concatenating axis "
											"must be the same, however, the array at index "
											+ std::to_string(i) + " had extent " + dims_n.str() +
											", where dimension " + std::to_string(j) + " was "
											"invalid.");
				}
			}

			if (axis < dims)
				new_dim += dims_n[axis];
		}

		if (dims == 0)
			throw std::domain_error("Cannot concatenate zero-dimensional arrays");

		if (axis >= dims)
			throw std::domain_error("Axis " + std::to_string(axis) + " is out of range for arrays "
									"with " + std::to_string(dims) + " dimensions");

		// Calculate the step for the result array.
		lr_int step = 1;
		for (lr_int d = axis; d < dims; d++)
			step *= dims_0[d];

		extent res_shape = arrays[0].get_extent();
		res_shape[axis] = new_dim;

		// We know res has a trivial stride with contiguous data, so
		// certain "cheats" can me made
		basic_ndarray<A> res(res_shape);

		// Iterate over arrays and stack them together
		lr_int offset = 0;
		for (const auto &arr : arrays)
		{
			lr_int idim = 0;

			const auto *array_extent = arr.get_extent().get_extent_alt();
			const auto *array_stride = arr.get_stride().get_stride_alt();

			const auto *res_extent = res.get_extent().get_extent_alt();
			const auto *res_stride = res.get_stride().get_stride_alt();

			auto *array_data = arr.get_data_start();
			auto *res_data = res.get_data_start() + offset;

			lr_int coord[LIBRAPID_MAX_DIMS]{};

			do
			{
				*res_data = *array_data;

				for (idim = 0; idim < dims; ++idim)
				{
					if (++coord[idim] == array_extent[idim])
					{
						coord[idim] = 0;
						array_data = array_data - (array_extent[idim] - 1) * array_stride[idim];
						res_data = res_data - (array_extent[idim] - 1) * res_stride[idim];
					}
					else
					{
						array_data = array_data + array_stride[idim];
						res_data = res_data + res_stride[idim];
						break;
					}
				}
			} while (idim < dims);

			offset += step;
		}

		return res;
	}

	template<typename A>
	LR_INLINE basic_ndarray<A> concatenate(const std::initializer_list<basic_ndarray<A>> &arrays,
										   lr_int axis = 0)
	{
		return concatenate(std::vector<basic_ndarray<A>>(arrays.begin(), arrays.end()), axis);
	}

	/**
	 * \rst
	 *
	 * Given a list of arrays, join them together along a new axis.
	 *
	 * Parameters
	 * ----------
	 *
	 * arrays: vector, list, tuple
	 *		The list of arrays to stack
	 * axis = 0: integer
	 *		The axis along which to stack the arrays
	 *
	 * Returns
	 * -------
	 * stacked: ndarray
	 *		An array with one more dimension than the input arrays
	 *
	 * .. Attention::
	 *		The arrays passed must all have exactly the same extent
	 *
	 * Examples
	 * --------
	 *
	 * .. code-block:: python
	 *
	 *		# Create the example arrays
	 *		first = librapid.from_data(
	 *			[1, 2, 3]
	 *		)
	 *
	 *		second = librapid.from_data(
	 *			[4, 5, 6]
	 *		)
	 *
	 *		# Stack the arrays and store the result
	 *		# (default axis is 0)
	 *		stacked = librapid.pack((first, second))
	 *
	 *		print(stacked)
	 *
	 *		"""
	 *		Gives:
	 *
	 *		[[1. 2. 3.]
	 * 		 [4. 5. 6.]]
	 *		"""
	 *
	 * .. code-block:: python
	 *
	 *		# Create the example arrays
	 *		first = librapid.from_data(
	 *			[[1, 2, 3],
	 *			 [4, 5, 6],
	 *			 [7, 8, 9]]
	 *		)
	 *
	 *		second = librapid.from_data(
	 *			[[10, 11, 12],
	 *			 [13, 14, 15],
	 *			 [16, 17, 18]]
	 *		)
	 *
	 *		# Stack the arrays and store the result
	 *		# (here, we are stacking on axis=1 -- along the columns)
	 *		stacked = librapid.stack((first, second), 1)
	 *
	 *		print(stacked)
	 *
	 *		"""
	 *		Gives:
	 *
	 *		[[[ 1.  2.  3.]
	 *		  [10. 11. 12.]]
	 *
	 *		 [[ 4.  5.  6.]
	 *		  [13. 14. 15.]]
	 *
	 *		 [[ 7.  8.  9.]
	 *		  [16. 17. 18.]]]
	 *		"""
	 *
	 * .. Hint::
	 *		The resulting array is contiguous in memory, meaning the resulting array
	 *		can be operated on very quickly and efficiently. (see ``ndarray.clone`` for
	 *		more information)
	 *
	 * \endrst
	 */
	template<typename A>
	LR_INLINE basic_ndarray<A> stack(const std::vector<basic_ndarray<A>> &arrays, lr_int axis = 0)
	{
		lr_int new_dim = arrays.size();

		if (new_dim == 0) return {};
		if (new_dim == 1) return arrays[0].clone();

		bool valid = true;
		const extent &dim_0 = arrays[0].get_extent();

		for (lr_int i = 1; i < new_dim; i++)
		{
			if (arrays[i].get_extent() != dim_0)
			{
				valid = false;
				break;
			}
		}

		if (!valid)
			throw std::domain_error("All arrays must have the same extent in order to pack them");

		lr_int dims = dim_0.ndim();
		lr_int new_shape[LIBRAPID_MAX_DIMS]{};

		for (lr_int dim = 0; dim < dims + 1; dim++)
		{
			if (dim < axis) new_shape[dim] = dim_0[dim];
			else if (dim == axis) new_shape[dim] = new_dim;
			else new_shape[dim] = dim_0[dim - 1];
		}

		// If accessing the last dimension, reshape the arrays
		// to ensure that the correct calculation is performed
		if (axis == dims)
		{
			lr_int tmp_shape[LIBRAPID_MAX_DIMS]{};

			for (lr_int dim = 0; dim < dims + 1; dim++)
			{
				if (dim == dims) tmp_shape[dim] = 1;
				else tmp_shape[dim] = dim_0[dim];
			}

			std::vector<basic_ndarray<A>> new_arrays;
			for (auto &arr : arrays)
				new_arrays.emplace_back(arr.reshaped(extent(tmp_shape, dims + 1)));

			return concatenate(new_arrays, axis).reshaped(extent(new_shape, dims + 1));
		}

		return concatenate(arrays, axis).reshaped(extent(new_shape, dims + 1));
	}

	template<typename A>
	LR_INLINE basic_ndarray<A> stack(const std::initializer_list<basic_ndarray<A>> &arrays,
										   lr_int axis = 0)
	{
		return stack(std::vector<basic_ndarray<A>>(arrays.begin(), arrays.end()), axis);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		minimum(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.minimum(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		minimum(const basic_ndarray<A> &x1, B x2)
	{
		return x1.minimum(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		minimum(A x1, const basic_ndarray<B> &x2)
	{
		return x2.minimum(x1);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		minimum(A x1, B x2)
	{
		return math::min(x1, x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		maximum(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.maximum(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		maximum(const basic_ndarray<A> &x1, B x2)
	{
		return x1.maximum(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		maximum(A x1, const basic_ndarray<B> &x2)
	{
		return x2.maximum(x1);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		maximum(A x1, B x2)
	{
		return math::max(x1, x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		less_than(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.less_than(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		less_than(const basic_ndarray<A> &x1, B x2)
	{
		return x1.less_than(x2);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		less_than(A x1, B x2)
	{
		return x1 < x2;
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		greater_than(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.greater_than(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		greater_than(const basic_ndarray<A> &x1, B x2)
	{
		return x1.greater_than(x2);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		greater_than(A x1, B x2)
	{
		return x1 > x2;
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		less_than_or_equal(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.less_than_or_equal(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		less_than_or_equal(const basic_ndarray<A> &x1, B x2)
	{
		return x1.less_than_or_equal(x2);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		less_than_or_equal(A x1, B x2)
	{
		return x1 <= x2;
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		greater_than_or_equal(const basic_ndarray<A> &x1, const basic_ndarray<B> &x2)
	{
		return x1.greater_than_or_equal(x2);
	}

	template<typename A, typename B>
	LR_INLINE basic_ndarray<typename std::common_type<A, B>::type>
		greater_than_or_equal(const basic_ndarray<A> &x1, B x2)
	{
		return x1.greater_than_or_equal(x2);
	}

	template<typename A, typename B>
	LR_INLINE typename std::common_type<A, B>::type
		greater_than_or_equal(A x1, B x2)
	{
		return x1 >= x2;
	}

	template<typename T, typename LAMBDA>
	LR_INLINE basic_ndarray<T> map(const basic_ndarray<T> &arr, LAMBDA func)
	{
		return arr.mapped(func);
	}

	template<typename T, typename A = lr_int>
	LR_INLINE basic_ndarray<T> sum(const basic_ndarray<T> &arr, A axis = AUTO)
	{
		return arr.sum(axis);
	}

	template<typename T, typename A = lr_int>
	LR_INLINE basic_ndarray<T> product(const basic_ndarray<T> &arr, A axis = AUTO)
	{
		return arr.product(axis);
	}

	template<typename T, typename A = lr_int>
	LR_INLINE basic_ndarray<T> mean(const basic_ndarray<T> &arr, A axis = AUTO)
	{
		return arr.mean(axis);
	}

	template<typename T>
	LR_INLINE basic_ndarray<T> abs(const basic_ndarray<T> &arr)
	{
		return arr.abs();
	}

	template<typename T>
	LR_INLINE basic_ndarray<T> square(const basic_ndarray<T> &arr)
	{
		return arr.square();
	}

	template<typename T>
	LR_INLINE basic_ndarray<T> sqrt(const basic_ndarray<T> &arr)
	{
		return arr.sqrt();
	}

	template<typename T, typename A = lr_int>
	LR_INLINE basic_ndarray<T> variance(const basic_ndarray<T> &arr, A axis = AUTO)
	{
		return arr.variance(axis);
	}

	using ndarray = basic_ndarray<double>;
	using ndarray_f = basic_ndarray<float>;
	using ndarray_i = basic_ndarray<int>;

	template<typename T>
	std::ostream &operator<<(std::ostream &os, const basic_ndarray<T> &arr)
	{
		return os << arr.str();
	}

	template<typename V = double>
	LR_INLINE basic_ndarray<V> from_data(V value)
	{
		return basic_ndarray<V>::from_data(value);
	}

	template<typename V = double>
	LR_INLINE basic_ndarray<V> from_data(const std::vector<V> &values)
	{
		basic_ndarray<V> res(extent({values.size()}));
		for (size_t i = 0; i < values.size(); i++)
			res.set_value(i, (V) values[i]);
		return res;
	}

	template<typename V = double>
	LR_INLINE basic_ndarray<V> from_data(const std::vector<std::vector<V>> &values)
	{
		std::vector<lr_int> size = utils::extract_size(values);
		auto res = basic_ndarray<V>(extent(size));
		for (size_t i = 0; i < values.size(); i++)
			res[i] = from_data(values[i]);
		return res;
	}
}

#endif // NDARRAY_BASIC_ARRAY