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

#include <librapid/math/rapid_math.hpp>
#include <librapid/ndarray/to_string.hpp>

// Define this if using a custom cblas interface.
// If it is not defined, the (slower) internal
// interface will be used.
#ifdef LIBRAPID_CBLAS
#include <cblas.h>
#endif

namespace librapid
{
	namespace ndarray
	{
		constexpr nd_int AUTO = -1;

		template<typename T>
		using nd_allocator = std::allocator<T>;

		template<typename T, class alloc = nd_allocator<T>,
			typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
			class basic_ndarray;

		template<typename A_T, typename B_T, typename B_A>
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator+(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator-(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
			nd_allocator<typename std::common_type<A_T, B_T>::type>>
			operator*(const A_T &val, const basic_ndarray<B_T, B_A> &arr);

		template<typename A_T, typename B_T, typename B_A>
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
			 * This is a function. It does something.
			 *
			 * Hello, World!
			 *
			 * \rst
			 *
			 * Hello. This is a title
			 * ======================
			 *
			 * 1. This
			 * 2. Is
			 * 3. A
			 * 4. Numbered
			 * 5. List
			 *
			 * +---------+------+--------+
			 * | This is |      | A      |
			 * |         +------+-----+--+
			 * |         | Very fancy |  |
			 * +=====+===+======+=====+==+
			 * |     | Table    |     |  |
			 * +-----+          +-----+--+
			 * |     |          |     |  |
			 * +-----+----------+-----+--+
			 *
			 * .. code-block:: Python
			 *
			 * 		print("Hello, World!")
			 *
			 * .. hint::
			 * 		This is a hint in a box! How cool is that!?
			 *
			 * \endrst
			 *
			 */
			void set_stride(const stride &s)
			{
				m_stride = s;
			}

			basic_ndarray() = default;

			template<typename V>
			basic_ndarray(const basic_extent<V> &size) : m_extent(size),
				m_stride(stride::from_extent(size.get_extent(), size.ndim())),
				m_extent_product(math::product(size.get_extent(), size.ndim()))
			{
				auto state = construct_new();

				if (state == errors::ALL_OK)
					return;

				if (state == errors::ARRAY_DIMENSIONS_TOO_LARGE)
					throw std::range_error("Too many dimensions in array. Maximum allowed is "
										   + std::to_string(ND_MAX_DIMS));
			}

			template<typename E, typename V>
			basic_ndarray(const basic_extent<E> &size, const V &val) : m_extent(size),
				m_stride(stride::from_extent(size.get_extent(), size.ndim())),
				m_extent_product(math::product(size.get_extent(), size.ndim()))
			{
				auto state = construct_new();

				if (state == errors::ALL_OK)
				{
					fill((T) val);
					return;
				}

				if (state == errors::ARRAY_DIMENSIONS_TOO_LARGE)
					throw std::range_error("Too many dimensions in array. Maximum allowed is "
										   + std::to_string(ND_MAX_DIMS));
			}

			basic_ndarray(const basic_ndarray<T> &arr) : m_extent(arr.m_extent),
				m_stride(arr.m_stride), m_origin_references(arr.m_origin_references),
				m_data_origin(arr.m_data_origin), m_data_start(arr.m_data_start),
				m_extent_product(arr.m_extent_product), m_origin_size(arr.m_origin_size),
				m_is_scalar(arr.m_is_scalar)
			{
				increment();
			}

			template<typename L>
			basic_ndarray(const std::initializer_list<L> &shape)
				: basic_ndarray(std::vector<L>(shape.begin(), shape.end()))
			{}

			template<typename L, typename V>
			basic_ndarray(const std::initializer_list<L> &shape, V value)
				: basic_ndarray(std::vector<L>(shape.begin(), shape.end()), (L) value)
			{}

			ND_INLINE basic_ndarray<T> &operator=(const basic_ndarray<T> &arr)
			{
				if (!(utils::check_ptr_match(m_extent.get_extent(),
					ndim(), arr.m_extent.get_extent(), arr.ndim())))
					throw std::domain_error("Invalid shape for array setting. " +
											m_extent.str() + " and " + arr.get_extent().str() +
											" are not equal.");

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
			ND_INLINE basic_ndarray &operator=(const V &other)
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

			ND_INLINE nd_int ndim() const
			{
				return (nd_int) m_extent.ndim();
			}

			ND_INLINE bool is_initialized() const
			{
				return m_origin_references != nullptr;
			}

			ND_INLINE bool is_scalar() const
			{
				return m_is_scalar;
			}

			ND_INLINE const extent &get_extent() const
			{
				return m_extent;
			}

			ND_INLINE const stride &get_stride() const
			{
				return m_stride;
			}

			ND_INLINE T *get_data_start() const
			{
				return m_data_start;
			}

			ND_INLINE basic_ndarray<T, alloc> operator[](nd_int index)
			{
				using non_const = typename std::remove_const<basic_ndarray<T, alloc>>::type;
				return (non_const) subscript(index);
			}

			ND_INLINE const basic_ndarray<T, alloc> operator[](nd_int index) const
			{
				return subscript(index);
			}

			template<typename I>
			ND_INLINE basic_ndarray<T, alloc> subarray(const std::vector<I> &index) const
			{
				// Validate the index

				if (index.size() != ndim())
					throw std::domain_error("Array with " + std::to_string(ndim()) +
											" dimensions requires " + std::to_string(index.size()) +
											" access elements");

				nd_int new_shape[ND_MAX_DIMS]{};
				nd_int new_stride[ND_MAX_DIMS]{};
				nd_int count = 0;

				T *new_start = m_data_start;

				for (nd_int i = 0; i < index.size(); i++)
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
				res.m_extent = extent(new_shape, count);
				res.m_extent_product = math::product(new_shape, count);
				res.m_is_scalar = count == 0;

				increment();

				return res;
			}

			ND_INLINE basic_ndarray<T, alloc> subarray(const std::initializer_list<nd_int> &index) const
			{
				return subarray(std::vector<nd_int>(index.begin(), index.end()));
			}

			template<class F>
			ND_INLINE void fill(const F &filler)
			{
				arithmetic::array_op(m_data_start, m_data_start, m_extent, m_stride, m_stride, [=]<typename V>(V x)
				{
					return filler;
				});
			}

			template<class F>
			ND_INLINE basic_ndarray<T, alloc> filled(const F &filler) const
			{
				basic_ndarray<T, alloc> res;
				res.construct_new(m_extent, m_stride);
				res.fill(filler);

				return res;
			}

			ND_INLINE basic_ndarray<T, alloc> clone() const
			{
				basic_ndarray<T, alloc> res(m_extent);

				res.m_origin_size = m_origin_size;
				res.m_is_scalar = m_is_scalar;

				if (!m_stride.is_trivial())
				{
					// Non-trivial stride, so use a more complicated accessing
					// method to ensure that the resulting array is contiguous
					// in memory for faster running times overall

					nd_int idim = 0;
					nd_int dims = ndim();

					const auto *__restrict _extent = m_extent.get_extent_alt();
					const auto *__restrict _stride_this = m_stride.get_stride_alt();
					auto *__restrict this_ptr = m_data_start;
					auto *__restrict res_ptr = res.get_data_start();

					nd_int coord[ND_MAX_DIMS]{};

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

			void set_value(nd_int index, T val)
			{
				m_data_start[index] = val;
			}

			template<typename B_T, typename B_A>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator+(const basic_ndarray<B_T, B_A> &other) const
			{
				return basic_ndarray<T, alloc>::
					array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a + b;
				});
			}

			template<typename B_T, typename B_A>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator-(const basic_ndarray<B_T, B_A> &other) const
			{
				return basic_ndarray<T, alloc>::
					array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a - b;
				});
			}

			template<typename B_T, typename B_A>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator*(const basic_ndarray<B_T, B_A> &other) const
			{
				return basic_ndarray<T, alloc>::
					array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a * b;
				});
			}

			template<typename B_T, typename B_A>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator/(const basic_ndarray<B_T, B_A> &other) const
			{
				return basic_ndarray<T, alloc>::
					array_array_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a / b;
				});
			}

			template<typename B_T>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator+(const B_T &other) const
			{
				return basic_ndarray<T, alloc>::
					array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a + b;
				});
			}

			template<typename B_T>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator-(const B_T &other) const
			{
				return basic_ndarray<T, alloc>::
					array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a - b;
				});
			}

			template<typename B_T>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator*(const B_T &other) const
			{
				return basic_ndarray<T, alloc>::
					array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a * b;
				});
			}

			template<typename B_T>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				operator/(const B_T &other) const
			{
				return basic_ndarray<T, alloc>::
					array_scalar_arithmetic(*this, other, []<typename T_a, typename T_b>(T_a a, T_b b)
				{
					return a / b;
				});
			}

			ND_INLINE basic_ndarray<T, alloc> operator-() const
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
			ND_INLINE void reshape(const basic_extent<O> &new_shape)
			{
				if (math::product(new_shape.get_extent(), new_shape.ndim()) != m_extent_product)
					throw std::length_error("Array sizes are different, so cannot reshape array. Shapes "
											+ m_extent.str() + " and " + new_shape.str() + " cannot be broadcast");

				if (!m_stride.is_trivial())
				{
					// Non-trivial stride, so this array will be deferenced and a new array
					// created in its place

					// This destroys the current array and replaces it!

					auto new_data = m_alloc.allocate(m_extent_product);

					nd_int idim = 0;
					nd_int dims = ndim();

					const auto *__restrict _extent = m_extent.get_extent_alt();
					const auto *__restrict _stride_this = m_stride.get_stride_alt();

					nd_int coord[ND_MAX_DIMS]{};

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

					m_origin_references = new std::atomic<nd_int>(1);

					m_origin_size = m_extent_product;
				}

				m_stride = stride::from_extent(std::vector<O>(new_shape.begin(),
											   new_shape.end()));
				m_extent = extent(new_shape);
			}

			template<typename O>
			ND_INLINE void reshape(const std::initializer_list<O> &new_shape)
			{
				reshape(std::vector<O>(new_shape.begin(), new_shape.end()));
			}

			template<typename O>
			ND_INLINE void reshape(const std::vector<O> &new_shape)
			{
				reshape(std::vector<O>(new_shape.begin(), new_shape.end()));
			}

			template<typename O>
			ND_INLINE basic_ndarray<T, alloc> reshaped(const basic_extent<O> &new_shape) const
			{
				auto res = create_reference();
				res.reshape(new_shape);
				return res;
			}

			template<typename O>
			ND_INLINE basic_ndarray<T, alloc> reshaped(const std::initializer_list<O> &new_shape) const
			{
				return reshaped(std::vector<O>(new_shape.begin(), new_shape.end()));
			}

			template<typename O>
			ND_INLINE basic_ndarray<T, alloc> reshaped(const std::vector<O> &new_shape) const
			{
				return reshaped(std::vector<O>(new_shape.begin(), new_shape.end()));
			}

			ND_INLINE void strip_front()
			{
				// Remove leading dimensions which are all 1

				nd_int strip_to = 0;
				for (nd_int i = 0; i < ndim(); i++)
					if (m_extent[i] == 1) strip_to++;
					else break;

				// Ensure arrays of shape [1, 1, ... 1] are not
				// completely erased
				if (strip_to == ndim())
					strip_to--;

				nd_int new_dims = ndim() - strip_to;

				nd_int new_extent[ND_MAX_DIMS]{};
				for (nd_int i = 0; i < new_dims; i++)
					new_extent[i] = m_extent[i + strip_to];

				nd_int new_stride[ND_MAX_DIMS]{};
				for (nd_int i = 0; i < new_dims; i++)
					new_stride[i] = m_stride[i + strip_to];

				m_stride = stride(new_stride, new_dims);
				m_extent = extent(new_extent, new_dims);
			}

			ND_INLINE void strip_back()
			{
				// Remove trailing dimensions which are all 1

				nd_int strip_to = ndim();
				for (nd_int i = ndim(); i > 0; i--)
					if (m_extent[i - 1] == 1) strip_to--;
					else break;

				// Ensure arrays of shape [1, 1, ... 1] are not
				// completely erased
				if (strip_to == 0)
					strip_to++;

				nd_int new_extent[ND_MAX_DIMS]{};
				for (nd_int i = 0; i < strip_to; i++)
					new_extent[i] = m_extent[i];

				nd_int new_stride[ND_MAX_DIMS]{};
				for (nd_int i = 0; i < strip_to; i++)
					new_stride[i] = m_stride[i];

				m_stride = stride(new_stride, strip_to);
				m_extent = extent(new_extent, strip_to);
			}

			ND_INLINE void strip()
			{
				strip_front();
				strip_back();
			}

			ND_INLINE basic_ndarray<T, alloc> stripped_front() const
			{
				auto res = create_reference();
				res.strip_front();
				return res;
			}

			ND_INLINE basic_ndarray<T, alloc> stripped_back() const
			{
				auto res = create_reference();
				res.strip_back();
				return res;
			}

			ND_INLINE basic_ndarray<T, alloc> stripped() const
			{
				auto res = create_reference();
				res.strip();
				return res;
			}

			template<typename O>
			ND_INLINE void transpose(const std::vector<O> &order)
			{
				// Validate the ordering
				if (order.size() != ndim())
				{
					std::string msg = "To transpose an array with " + std::to_string(ndim()) + " dimensions, "
						+ std::to_string(ndim()) + " indices are required, but only " +
						std::to_string(order.size()) + " were supplied";
					throw std::domain_error(msg);
				}

				bool valid = true;
				std::vector<O> missing;
				for (nd_int i = 0; i < ndim(); i++)
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
					for (nd_int i = 0; i < m_stride.ndim(); i++)
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

			ND_INLINE void transpose()
			{
				std::vector<nd_int> order(ndim());
				for (nd_int i = 0; i < ndim(); i++)
					order[i] = ndim() - i - 1;
				transpose(order);
			}

			template<typename O>
			ND_INLINE void transpose(const std::initializer_list<O> &order)
			{
				transpose(std::vector<O>(order.begin(), order.end()));
			}

			template<typename O>
			ND_INLINE basic_ndarray<T, alloc> transposed(const std::vector<O> &order) const
			{
				auto res = create_reference();
				res.transpose(order);
				return res;
			}

			template<typename O>
			ND_INLINE basic_ndarray<T, alloc> transposed(const std::initializer_list<O> &order) const
			{
				return transposed(order);
			}

			ND_INLINE basic_ndarray<T, alloc> transposed() const
			{
				auto res = create_reference();
				res.transpose();
				return res;
			}

			template<typename B_T, typename B_A>
			ND_INLINE basic_ndarray<typename std::common_type<T, B_T>::type,
				nd_allocator<typename std::common_type<T, B_T>::type>>
				dot(const basic_ndarray<B_T, B_A> &other) const
			{
				using R_T = typename std::common_type<T, B_T>::type;

				const auto &o_e = other.get_extent();
				if (utils::check_ptr_match(o_e.get_extent(), o_e.ndim(),
					utils::sub_vector(m_extent.get_extent(), m_extent.ndim(), 1),
					true))
				{
					// Matrix-Vector product
					nd_int res_shape[ND_MAX_DIMS]{};
					res_shape[0] = m_extent[0];

					nd_int dims = other.ndim();
					for (nd_int i = 1; i < dims; i++)
						res_shape[i] = other.get_extent()[i];

					auto res = basic_ndarray<R_T>(extent(res_shape, dims));
					for (nd_int i = 0; i < m_extent[0]; i++)
						res[i] = subscript(i).dot(other);

					return res;
				}

				if (ndim() != other.ndim())
					throw std::range_error("Cannot compute dot product on arrays with " +
										   m_extent.str() + " and " + other.get_extent().str());

				nd_int dims = ndim();

				switch (dims)
				{
					case 1:
						{
							if (m_extent[0] != other.get_extent()[0])
								throw std::range_error("Cannot compute dot product with arrays with " +
													   m_extent.str() + " and " + other.get_extent().str());

							// Vector product
							basic_ndarray<R_T, nd_allocator<R_T>> res(extent({1}));
							res.m_is_scalar = true;

						#ifndef LIBRAPID_CBLAS
							R_T *r = res.get_data_start();
							T *m = m_data_start;
							B_T *o = other.get_data_start();
							nd_int lda = m_stride[0];
							nd_int ldb = other.get_stride()[0];
							*r = 0;
							for (nd_int i = 0; i < m_extent_product; i++)
								*r += m[i * lda] * o[i * ldb];
						#else
							*res.get_data_start() = cblas_ddot(m_extent_product, m_data_start, m_stride[0],
															   other.get_data_start(), other.get_stride()[0]);
						#endif // LIBRAPID_CBLAS

							return res;
						}
					case 2:
						{
							if (m_extent[1] != other.get_extent()[0])
								throw std::range_error("Cannot compute dot product with arrays with " +
													   m_extent.str() + " and " + other.get_extent().str());

						#ifndef LIBRAPID_CBLAS
							const basic_ndarray<T, alloc> tmp_other = other.transposed_matrix();

							nd_int M = get_extent()[0];
							nd_int N = get_extent()[1];
							nd_int K = tmp_other.get_extent()[1];
							nd_int K_prime = other.get_extent()[1];

							auto res = basic_ndarray<R_T>(extent{M, K_prime});

							T *a = m_data_start;
							B_T *b = tmp_other.get_data_start();
							R_T *c = res.get_data_start();

							nd_int lda = m_stride[0], fda = m_stride[1];
							nd_int ldb = tmp_other.get_stride()[0], fdb = tmp_other.get_stride()[1];
							nd_int ldc = res.get_stride()[0], fdc = res.get_stride()[1];

							nd_int index_a, index_b, index_c;

							// Only run in parallel if arrays are smaller than a
							// given size. Running in parallel on smaller matrices
							// will result in slower code. Note, a branch is used
							// in preference to #pragma omp ... if (...) because
							// that requires runtime evaluation of a condition to
							// set up threads, which adds a significant overhead
							if (M * N * K < 25000)
							{
								for (nd_int outer = 0; outer < M; ++outer)
								{
									for (nd_int inner = 0; inner < K_prime; ++inner)
									{
										index_c = outer * ldc + inner * fdc;
										c[index_c] = 0;

										for (nd_int k = 0; k < N; k++)
										{
											index_a = outer * lda + k * fda;
											index_b = inner * ldb + k * fdb;

											c[index_c] += a[index_a] * b[index_b];
										}
									}
								}
							}
							else
							{
							#pragma omp parallel for shared(a, b, c, M, N, K, K_prime, lda, ldb, ldc, fda, fdb, fdc) private(index_a, index_b, index_c) default(none) num_threads(ND_NUM_THREADS)
								for (nd_int outer = 0; outer < M; ++outer)
								{
									for (nd_int inner = 0; inner < K_prime; ++inner)
									{
										index_c = outer * ldc + inner * fdc;
										c[index_c] = 0;

										for (nd_int k = 0; k < N; k++)
										{
											index_a = outer * lda + k * fda;
											index_b = inner * ldb + k * fdb;

											c[index_c] += a[index_a] * b[index_b];
										}
									}
								}
							}

							return res;
						#else

							const auto M = m_extent[0];
							const auto N = m_extent[1];
							const auto K = other.get_extent()[1];

							const R_T alpha = 1.0;
							const R_T beta = 0.0;

							auto res = basic_ndarray<R_T>(extent{M, K});

							const auto transA = m_stride[0] == N ? CblasNoTrans : CblasTrans;
							const auto transB = other.get_stride()[0] == K ? CblasNoTrans : CblasTrans;

							const auto lda = N;
							const auto ldb = K;
							const auto ldc = K;

							auto *__restrict a = m_data_start;
							auto *__restrict b = other.get_data_start();
							auto *__restrict c = res.get_data_start();

							cblas_dgemm(CblasRowMajor, transA, transB, M, K, N,
										alpha, a, lda, b, ldb, beta, c, ldc);

							return res;
						#endif // LIBRAPID_CBLAS
						}
					default:
						{
							// Check the arrays are valid
							if (m_extent[ndim() - 1] != other.get_extent()[other.ndim() - 2])
							{
								throw std::range_error("Cannot compute dot product with arrays with " +
													   m_extent.str() + " and " + other.get_extent().str());
							}

							// Create the new array dimensions
							nd_int new_extent[ND_MAX_DIMS]{};

							// Initialize the new dimensions
							for (nd_int i = 0; i < ndim() - 1; i++) new_extent[i] = m_extent[i];
							for (nd_int i = 0; i < other.ndim() - 2; i++) new_extent[i + ndim()] = other.get_extent()[i];
							new_extent[ndim() + other.ndim() - 4] = other.get_extent()[other.ndim() - 3];
							new_extent[ndim() + other.ndim() - 3] = other.get_extent()[other.ndim() - 1];

							auto res = basic_ndarray<R_T, nd_allocator<R_T>>(extent(new_extent, ndim() + other.ndim() - 2));
							R_T *__restrict res_ptr = res.get_data_start();

							nd_int idim = 0;
							nd_int dims = res.ndim();

							const auto *__restrict _extent = res.get_extent().get_extent();
							const auto *__restrict _stride = res.get_stride().get_stride();

							nd_int coord[ND_MAX_DIMS]{};

							std::vector<nd_int> lhs_index(ndim());
							std::vector<nd_int> rhs_index(other.ndim());

							do
							{
								// Extract the index for the lhs
								for (nd_int i = 0; i < ndim() - 1; i++)
									lhs_index[i] = coord[i];
								lhs_index[ndim() - 1] = AUTO;

								// Extract the index for the rhs
								for (nd_int i = 0; i < other.ndim() - 2; i++)
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

			std::string str(nd_int start_depth = 0) const
			{
				const auto *__restrict extent_data = m_extent.get_extent();

				if (!is_initialized())
					return "[NONE]";

				if (m_is_scalar)
					return to_string::format_numerical(m_data_start[0]).str;

				std::vector<to_string::str_container> formatted(m_extent_product, {"", 0});
				nd_int longest_integral = 0;
				nd_int longest_decimal = 0;

				// General checks
				bool strip_middle = false;
				if (m_extent_product > 1000)
					strip_middle = true;

				// Edge case
				if (ndim() == 2 && extent_data[1] == 1)
					strip_middle = false;

				nd_int idim = 0;
				nd_int dimensions = ndim();
				nd_int index = 0;
				nd_int data_index = 0;
				auto coord = new nd_int[dimensions];
				memset(coord, 0, sizeof(nd_int) * dimensions);

				std::vector<nd_int> tmp_extent(dimensions);
				std::vector<nd_int> tmp_stride(dimensions);
				for (nd_int i = 0; i < dimensions; i++)
				{
					tmp_stride[dimensions - i - 1] = m_stride.get_stride()[i];
					tmp_extent[dimensions - i - 1] = m_extent.get_extent()[i];
				}

				do
				{
					bool skip = false;
					for (nd_int i = 0; i < dimensions; i++)
					{
						if (strip_middle &&
							(coord[i] > 3 && coord[i] < extent_data[i] - 3))
						{
							skip = true;
							break;
						}
					}

					if (!skip)
					{
						formatted[index] = to_string::format_numerical(m_data_start[data_index]);

						if (formatted[index].decimal_point > longest_integral)
							longest_integral = formatted[index].decimal_point;

						auto &format_tmp = formatted[index];
						if (format_tmp.str.length() >= format_tmp.decimal_point &&
							format_tmp.str.length() - format_tmp.decimal_point > longest_decimal)
							longest_decimal = format_tmp.str.length() - format_tmp.decimal_point;
					}

					index++;

					for (idim = 0; idim < dimensions; ++idim)
					{
						if (++coord[idim] == tmp_extent[idim])
						{
							coord[idim] = 0;
							data_index -= (tmp_extent[idim] - 1) * tmp_stride[idim];
						}
						else
						{
							data_index += tmp_stride[idim];
							break;
						}
					}
				} while (idim < dimensions);

				delete[] coord;

				std::vector<std::string> adjusted(formatted.size(), "");

				for (nd_int i = 0; i < formatted.size(); i++)
				{
					if (formatted[i].str.empty())
						continue;

					const auto &term = formatted[i];
					nd_int decimal = (term.str.length() - term.decimal_point - 1);

					auto tmp = std::string((nd_int) (longest_integral - (T) term.decimal_point), ' ')
						+ term.str + std::string((nd_int) (longest_decimal - decimal), ' ');
					adjusted[i] = tmp;
				}

				std::vector<nd_int> extent_vector(ndim());
				for (nd_int i = 0; i < ndim(); i++)
					extent_vector[i] = extent_data[i];

				auto res = to_string::to_string(adjusted, extent_vector, 1 + start_depth, strip_middle);

				return res;
			}

		private:
			ND_INLINE errors construct_new()
			{
				if (ndim() > ND_MAX_DIMS)
					return errors::ARRAY_DIMENSIONS_TOO_LARGE;

				m_data_start = m_alloc.allocate(m_extent_product);
				m_origin_size = m_extent_product;
				m_data_origin = m_data_start;

				m_origin_references = new std::atomic<nd_int>(1);

				return errors::ALL_OK;
			}

			template<typename E, typename S>
			ND_INLINE errors construct_new(const basic_extent<E> &e, const basic_stride<S> &s)
			{
				m_extent = e;
				m_stride = s;

				if (ndim() > ND_MAX_DIMS)
					return errors::ARRAY_DIMENSIONS_TOO_LARGE;

				m_extent_product = math::product(m_extent.get_extent(), ndim());

				m_data_start = m_alloc.allocate(m_extent_product);
				m_origin_size = m_extent_product;

				m_data_origin = m_data_start;
				m_origin_references = new std::atomic<nd_int>(1);

				return errors::ALL_OK;
			}

			template<typename E, typename S>
			ND_INLINE errors construct_hollow(const basic_extent<E> &e, const basic_stride<S> &s)
			{
				m_extent = e;
				m_stride = s;

				if (ndim() > ND_MAX_DIMS)
					return errors::ARRAY_DIMENSIONS_TOO_LARGE;

				m_extent_product = math::product(m_extent.get_extent(), m_extent.ndim());
				m_origin_size = m_extent_product;

				return errors::ALL_OK;
			}

			ND_INLINE basic_ndarray<T, alloc> create_reference() const
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

			ND_INLINE void increment() const
			{
				++(*m_origin_references);
			}

			ND_INLINE void decrement()
			{
				--(*m_origin_references);

				if ((*m_origin_references) == 0)
				{
					m_alloc.deallocate(m_data_origin, m_origin_size);
					delete m_origin_references;
				}
			}

			ND_INLINE const basic_ndarray<T, alloc> subscript(nd_int index) const
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
				nd_int dims = ndim();

				nd_int new_extent[ND_MAX_DIMS]{};
				nd_int new_stride[ND_MAX_DIMS]{};

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
					memcpy(new_extent, m_extent.get_extent() + 1, sizeof(nd_int) * (ND_MAX_DIMS - 1));
					memcpy(new_stride, m_stride.get_stride() + 1, sizeof(nd_int) * (ND_MAX_DIMS - 1));

					res.construct_hollow(extent(new_extent, m_extent.ndim() - 1),
										 stride(new_stride, m_stride.ndim() - 1));

					res.m_is_scalar = false;
				}

				increment();
				return res;
			}

			ND_INLINE basic_ndarray<T, alloc> transposed_matrix() const
			{
				if (ndim() != 2)
					throw std::domain_error("Cannot matrix transpose array with shape " + m_extent.str());

				basic_ndarray<T, alloc> res(extent{m_extent[1], m_extent[0]});
				nd_int lda = m_stride[0], fda = m_stride[1];
				nd_int scal = m_extent[1];

				for (nd_int i = 0; i < m_extent[0]; i++)
					for (nd_int j = 0; j < m_extent[1]; j++)
						res.set_value(i + j * scal, m_data_start[j * fda + i * lda]);

				return res;
			}

			template<typename A_T, typename A_A, typename B_T, typename B_A, typename LAMBDA>
			static ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
				nd_allocator<typename std::common_type<A_T, B_T>::type>>
				array_array_arithmetic(const basic_ndarray<A_T, A_A> &a,
									   const basic_ndarray<B_T, B_A> &b, LAMBDA op)
			{
				using C = typename std::common_type<A_T, B_T>::type;
				using R = nd_allocator<typename std::common_type<A_T, B_T>::type>;

				auto mode = broadcast::calculate_arithmetic_mode(a.get_extent().get_extent(),
																 a.ndim(), b.get_extent().get_extent(), b.ndim());

				if (mode == (nd_int) -1)
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
							//  > "Row by row" addition

							auto res = basic_ndarray<C, R>(a.get_extent());

							for (nd_int i = 0; i < a.get_extent()[0]; i++)
								res[i] = op(a[i], b);

							return res;
						}
					case 4:
						{
							// Cases:
							//  > Reverse "row by row" addition

							auto res = basic_ndarray<C, R>(b.get_extent());

							for (nd_int i = 0; i < b.get_extent()[0]; i++)
								res[i] = op(a, b[i]);

							return res;
						}
					case 5:
						{
							// Cases
							//  > Grid addition

							extent res_shape(b.ndim() + 1);
							for (nd_int i = 0; i < b.ndim(); i++)
								res_shape[i] = a.get_extent()[i];
							res_shape[b.ndim()] = b.get_extent()[b.ndim() - 1];

							auto res = basic_ndarray<C, R>(res_shape);

							for (nd_int i = 0; i < res_shape[0]; i++)
								res[i] = op(a[i], b);

							return res;
						}
					case 6:
						{
							// Cases
							//  > Reverse grid addition

							extent res_shape(a.ndim() + 1);
							for (nd_int i = 0; i < a.ndim(); i++)
								res_shape[i] = b.get_extent()[i];
							res_shape[a.ndim()] = a.get_extent()[a.ndim() - 1];

							auto res = basic_ndarray<C, R>(res_shape, a.ndim() + 1);

							for (nd_int i = 0; i < res_shape[0]; i++)
								res[i] = op(a, b[i]);

							return res;
						}
					case 7:
						{
							// Cases
							//  > "Column by column" addition

							if (a.ndim() == 2)
								return op(a.transposed().stripped(), b.transposed()).transposed();

							nd_int new_extent[ND_MAX_DIMS]{};
							nd_int i = 0;
							for (; i < a.ndim() - 1; i++)
								new_extent[i] = a.get_extent()[i];
							new_extent[i] = b.get_extent()[1];

							auto res = basic_ndarray<C, R>(extent(new_extent, a.ndim()));
							auto tmp_a = a;
							auto tmp_b = b.transposed();

							for (nd_int i = 0; i < res.get_extent()[0]; i++)
								res[i] = op(tmp_a[i].transposed().stripped(), tmp_b).transposed();

							return res;
						}
					case 8:
						{
							// Cases:
							// Check for reverse "column by column" addition

							if (b.ndim() == 2)
								return op(a.transposed(), b.transposed().stripped()).transposed();

							nd_int new_extent[ND_MAX_DIMS]{};
							nd_int i = 0;
							for (; i < b.ndim() - 1; i++)
								new_extent[i] = b.get_extent()[i];
							new_extent[i] = a.get_extent()[1];

							auto res = basic_ndarray<C, R>(extent(new_extent, b.ndim()));
							auto tmp_a = a.transposed();
							auto tmp_b = b;

							for (nd_int i = 0; i < res.get_extent()[0]; i++)
								res[i] = op(tmp_a, tmp_b[i].transposed().stripped()).transposed();

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
			static ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
			static ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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

		private:

			T *m_data_origin = nullptr;
			std::atomic<nd_int> *m_origin_references = nullptr;

			nd_int m_origin_size = 0;

			T *m_data_start = nullptr;

			stride m_stride;
			extent m_extent;
			nd_int m_extent_product = 0;
			bool m_is_scalar = false;

			_alloc m_alloc = alloc();
		};

		template<typename A_T, typename B_T, typename B_A>
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
		ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
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
		 *\rst
		 *
		 *.. Hint::
		 *	This function is mostly for compatibility
		 *	with the C# port of the library, as the
		 *	C++ and Python libraries support overloaded
		 *	operators.
		 *
		 *Add two values together and return the result.
		 *
		 *The input values can be any type that supports
		 *addition. In general, the return type will be
		 *the higher precision of the two input types,
		 *or an n-dimensional array if one is passed.
		 *
		 *Parameters
		 *----------
		 *
		 *addend1: any
		 *	The left-hand side of the addition operation
		 *addend2: any
		 *	The right-hand side of the addition operation
		 *
		 *Returns
		 *-------
		 *
		 *sum: any
		 *	The result of the addition calculation
		 *
		 *\endrst
		 */
		template<typename T_A, typename T_B>
		ND_INLINE auto add(const T_A &addend1, const T_B &addend2)
		{
			return addend1 + addend2;
		}

		/**
		 * \rst
		 *
		 * .. Hint::
		 *		This function is mostly for compatibility
		 *		with the C# port of the library, as the
		 *		C++ and Python libraries support overloaded
		 *		operators.
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
		ND_INLINE auto sub(const T_A &minuend, const T_B &subtrahend)
		{
			return minuend - subtrahend;
		}

		/**
		 * \rst
		 *
		 * .. Hint::
		 *		This function is mostly for compatibility
		 *		with the C# port of the library, as the
		 *		C++ and Python libraries support overloaded
		 *		operators.
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
		ND_INLINE auto mul(const T_A &factor1, const T_B &factor2)
		{
			return factor1 * factor2;
		}

		/**
		 * \rst
		 *
		 * .. Hint::
		 *		This function is mostly for compatibility
		 *		with the C# port of the library, as the
		 *		C++ and Python libraries support overloaded
		 *		operators.
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
		ND_INLINE auto div(const T_A &dividend, const T_B &divisor)
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
		ND_INLINE basic_ndarray<A_T, A_A> exp(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> sin(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> cos(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> tan(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> asin(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> acos(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> atan(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> sinh(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> cosh(const basic_ndarray<A_T, A_A> &arr)
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
		ND_INLINE basic_ndarray<A_T, A_A> tanh(const basic_ndarray<A_T, A_A> &arr)
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

		template<typename T, class alloc, typename O = nd_int>
		ND_INLINE basic_ndarray<T, alloc> reshape(const basic_ndarray<T, alloc> &arr, const basic_extent<O> &new_shape)
		{
			return arr.reshaped(new_shape);
		}

		using ndarray = basic_ndarray<double>;
		using ndarray_f = basic_ndarray<float>;
		using ndarray_i = basic_ndarray<int>;
	}
}

#endif // NDARRAY_BASIC_ARRAY