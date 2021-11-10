#ifndef LIBRAPID_EXTENT_STRIDE_PROXY
#define LIBRAPID_EXTENT_STRIDE_PROXY

#include <librapid/config.hpp>
#include <librapid/array/iterators.hpp>
#include <vector>

namespace librapid
{
	namespace imp
	{
		template<typename T>
		class Proxy
		{
		public:
			Proxy() = default;

			Proxy(T *val, int64_t index) : m_val(val), m_index(index)
			{}

			template<typename V>
			inline Proxy<T> &operator=(const V &value)
			{
				*m_val[m_index] = value;
				m_val->update();
				return *this;
			}

			template<typename V>
			inline T operator+(const V &value)
			{
				return *m_val[m_index] + value;
			}

			template<typename V>
			inline T operator-(const V &value)
			{
				return *m_val[m_index] - value;
			}

			template<typename V>
			inline T operator*(const V &value)
			{
				return *m_val[m_index] * value;
			}

			template<typename V>
			inline T operator/(const V &value)
			{
				return *m_val[m_index] / value;
			}

			template<typename V>
			inline Proxy<T> &operator+=(const V &value)
			{
				*m_val[m_index] += value;
				return *this;
			}

			template<typename V>
			inline Proxy<T> &operator-=(const V &value)
			{
				*m_val[m_index] -= value;
				return *this;
			}

			template<typename V>
			inline Proxy<T> &operator*=(const V &value)
			{
				*m_val[m_index] *= value;
			}

			template<typename V>
			inline Proxy<T> &operator/=(const V &value)
			{
				*m_val[m_index] /= value;
				return *this;
			}

			template<typename V>
			inline operator V() const
			{
				return *m_val[m_index];
			}

			inline std::string str() const
			{
				return std::to_string(*m_val[m_index]);
			}

		private:
			T *m_val;
			int64_t m_index = -1;
		};

		template<typename T>
		inline std::ostream &operator<<(std::ostream &os, const Proxy<T> &extent)
		{
			return os << extent.str();
		}
	}
}

#endif // LIBRAPID_EXTENT_STRIDE_PROXY