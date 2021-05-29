#ifndef NDARRAY_BASIC_ARRAY
#define NDARRAY_BASIC_ARRAY

#include <memory>
#include <type_traits>

#include <string>
#include <sstream>
#include <ostream>

#include <vector>
#include <algorithm>

#include <librapid/ndarray/to_string.hpp>

namespace ndarray
{
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

		basic_ndarray()
		{}

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
			: basic_ndarray(std::vector<L>(shape.begin(), shape.end()), value)
		{}

		ND_INLINE basic_ndarray<T> &operator=(const basic_ndarray<T> &arr)
		{
			if (!arr.is_initialized())
				return *this;

			if (!(utils::check_ptr_match(m_extent.get_extent(),
				ndim(), arr.m_extent.get_extent(), arr.ndim())))
				throw std::length_error("Invalid shape for array setting. Dimensions are not equal.");

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
				throw std::runtime_error("Cannot set non-scalar value to a scalar");

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

		template<class F>
		ND_INLINE void fill(const F &filler)
		{
			nd_int index = 0;
			constexpr nd_int inc = 4;

			if (m_extent_product > 3)
			{
				for (; index < m_extent_product - 3; index += inc)
				{
					m_data_start[index + 0] = (T) filler;
					m_data_start[index + 1] = (T) filler;
					m_data_start[index + 2] = (T) filler;
					m_data_start[index + 3] = (T) filler;
				}
			}

			for (; index < m_extent_product; index++)
				m_data_start[index] = (T) filler;
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
				// Non-trivial stride, so this array will be deferenced and a new array
				// created in its place

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
		ND_INLINE void reshape(const std::vector<O> &order)
		{
			if (math::product(order) != m_extent_product)
				throw std::length_error("Array sizes are different, so cannot reshape array. Shapes "
										+ m_extent.str() + " and " + extent(order).str() + " cannot be broadcast");

			if (!m_stride.is_trivial())
			{
				// Non-trivial stride, so this array will be deferenced and a new array
				// created in its place

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

				decrement();

				m_data_origin = new_data;
				m_data_start = new_data;

				m_origin_references = new nd_int(1);

				m_origin_size = m_extent_product;
			}

			m_stride = stride::from_extent(order);
			m_extent = extent(order);
		}

		template<typename O>
		ND_INLINE void reshape(const std::initializer_list<O> &order)
		{
			reshape(std::vector<O>(order.begin(), order.end()));
		}

		template<typename O>
		ND_INLINE basic_ndarray<T, alloc> reshaped(const std::vector<O> &order) const
		{
			auto res = create_reference();
			res.reshape(order);
			return res;
		}

		template<typename O>
		ND_INLINE basic_ndarray<T, alloc> reshaped(const std::initializer_list<O> &order)
		{
			return reshaped(std::vector<O>(order.begin(), order.end()));
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
			m_origin_references = new nd_int(1);

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
			m_origin_references = new nd_int(1);

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
			res.m_extent = m_extent;
			res.m_stride = m_stride;

			res.m_origin_size = m_origin_size;
			res.m_origin_references = m_origin_references;

			res.m_data_origin = m_data_origin;
			res.m_data_start = m_data_start;

			res.m_extent_product = m_extent_product;

			increment();
			return res;
		}

		ND_INLINE void increment() const
		{
			(*m_origin_references)++;
		}

		ND_INLINE void decrement()
		{
			(*m_origin_references)--;

			if (*m_origin_references == 0)
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
		nd_int *m_origin_references = nullptr;
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
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr, []<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a + b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator-(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr, []<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a - b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator*(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr, []<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a * b;
		});
	}

	template<typename A_T, typename B_T, typename B_A>
	ND_INLINE basic_ndarray<typename std::common_type<A_T, B_T>::type,
		nd_allocator<typename std::common_type<A_T, B_T>::type>>
		operator/(const A_T &val, const basic_ndarray<B_T, B_A> &arr)
	{
		return basic_ndarray<B_T, B_A>::scalar_array_arithmetic(val, arr, []<typename T_a, typename T_b>(T_a a, T_b b)
		{
			return a / b;
		});
	}

	using ndarray = basic_ndarray<double>;
	using ndarray_f = basic_ndarray<float>;
	using ndarray_i = basic_ndarray<int>;
}

#endif // NDARRAY_BASIC_ARRAY
