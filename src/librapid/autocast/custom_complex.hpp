/**
 * An inherited std::complex type to allow complex types
 * in the autocast library (not enough casting methods exist,
 * so new ones need to be implemented)
 */

#ifndef LIBRAPID_CUSTOM_COMPLEX
#define LIBRAPID_CUSTOM_COMPLEX

#ifndef _CONSTEXPR20
#define _CONSTEXPR20
#define UNDEF_CONTEXPR20
#endif

#include <librapid/stringmethods/format_number.hpp>

namespace librapid
{
	template <class T>
	class complex
	{
	public:
		complex(const T &real_val = T(), const T &imag_val = T())
			: m_real(real_val), m_imag(imag_val)
		{}

		LR_INLINE complex &operator=(const T &val)
		{
			m_real = val;
			m_imag = 0;
			return *this;
		}

		template <class V>
		LR_INLINE complex(const complex<V> &other)
			: complex(static_cast<T>(other.real()), static_cast<T>(other.imag()))
		{}

		template <class V>
		LR_INLINE complex &operator=(const complex<V> &other)
		{
			m_real = static_cast<T>(other.real());
			m_imag = static_cast<T>(other.imag());
			return *this;
		}

		LR_INLINE complex copy() const
		{
			return complex<T>(m_real, m_imag);
		}

		LR_INLINE complex operator-() const
		{
			return complex<T>(-m_real, -m_imag);
		}

		template<typename V>
		LR_INLINE complex operator+(const V &other) const
		{
			return complex<T>(m_real + other, m_imag);
		}

		template<typename V>
		LR_INLINE complex operator-(const V &other) const
		{
			return complex<T>(m_real - other, m_imag);
		}

		template<typename V>
		LR_INLINE complex operator*(const V &other) const
		{
			return complex<T>(m_real * other, m_imag * other);
		}

		template<typename V>
		LR_INLINE complex operator/(const V &other) const
		{
			return complex<T>(m_real / other, m_imag / other);
		}

		template<typename V>
		LR_INLINE complex &operator+=(const V &other)
		{
			m_real += other;
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator-=(const V &other)
		{
			m_real -= other;
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator*=(const V &other)
		{
			m_real *= other;
			m_imag *= other;
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator/=(const V &other)
		{
			m_real /= other;
			m_imag /= other;
			return *this;
		}

		template<typename V>
		LR_INLINE complex operator+(const complex<V> &other) const
		{
			return complex(m_real + other.real(),
						   m_imag + other.imag());
		}

		template<typename V>
		LR_INLINE complex operator-(const complex<V> &other) const
		{
			return complex(m_real - other.real(),
						   m_imag - other.imag());
		}

		template<typename V>
		LR_INLINE complex operator*(const complex<V> &other) const
		{
			return complex((m_real * other.real()) - (m_imag * other.imag()),
						   (m_real * other.imag()) + (m_imag * other.real()));
		}

		template<typename V>
		LR_INLINE complex operator/(const complex<V> &other) const
		{
			return complex((m_real * other.real()) + (m_imag * other.imag()) /
						   ((other.real() * other.real()) + (other.imag() * other.imag())),
						   (m_real * other.real()) - (m_imag * other.imag()) /
						   ((other.real() * other.real()) + (other.imag() * other.imag())));
		}

		template<typename V>
		LR_INLINE complex &operator+=(const complex<V> &other)
		{
			m_real = m_real + other.real();
			m_imag = m_imag + other.imag();
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator-=(const complex<V> &other)
		{
			m_real = m_real - other.real();
			m_imag = m_imag - other.imag();
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator*=(const complex<V> &other)
		{
			m_real = (m_real * other.real()) - (m_imag * other.imag());
			m_imag = (m_real * other.imag()) + (imag() * other.real());
			return *this;
		}

		template<typename V>
		LR_INLINE complex &operator/=(const complex<V> &other)
		{

			m_real = (m_real * other.real()) + (m_imag * other.imag()) /
				((other.real() * other.real()) + (other.imag() * other.imag()));
			m_imag = (m_real * other.real()) - (m_imag * other.imag()) /
				((other.real() * other.real()) + (other.imag() * other.imag()));
			return *this;
		}

		template<typename V>
		LR_INLINE bool operator==(const complex<V> &other) const
		{
			return m_real == other.real() && m_imag == other.imag();
		}

		template<typename V>
		LR_INLINE bool operator!=(const complex<V> &other) const
		{
			return !(*this == other);
		}

		LR_INLINE T mag() const
		{
			return std::sqrt(m_real * m_real + m_imag * m_imag);
		}

		LR_INLINE T angle() const
		{
			return std::atan2(m_real, m_imag);
		}

		LR_INLINE complex<T> log() const
		{
			return complex<T>(std::log(mag()), angle());
		}

		LR_INLINE complex<T> conjugate() const
		{
			return complex<T>(m_real, -m_imag);
		}

		LR_INLINE complex<T> reciprocal() const
		{
			return complex<T>((m_real) / (m_real * m_real + m_imag * m_imag),
							  -(m_imag) / (m_real * m_real + m_imag * m_imag));
		}

		LR_INLINE const T &real() const
		{
			return m_real;
		}

		LR_INLINE T &real()
		{
			return m_real;
		}

		LR_INLINE const T &imag() const
		{
			return m_imag;
		}

		LR_INLINE T &imag()
		{
			return m_imag;
		}

		template<typename V>
		LR_INLINE operator V() const
		{
			return (V) m_real;
		}

		template<typename V>
		LR_INLINE operator std::complex<V>() const
		{
			return std::complex<V>(m_real, m_imag);
		}

		LR_INLINE std::string str() const
		{
			std::string res;
			res += format_number(m_real);
			if (m_imag >= 0)
				res += "+";
			res += format_number(m_imag);
			res += "j";
			return res;
		}

	private:
		T m_real = 0;
		T m_imag = 0;
	};

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE complex<B> operator+(const A &a, const complex<B> &b)
	{
		return complex<B>(a) + b;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE complex<B> operator-(const A &a, const complex<B> &b)
	{
		return complex<B>(a) - b;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE complex<B> operator*(const A &a, const complex<B> &b)
	{
		return complex<B>(a) * b;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE complex<B> operator/(const A &a, const complex<B> &b)
	{
		return complex<B>(a) / b;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE A &operator+=(A &a, const complex<B> &b)
	{
		a += b.real();
		return a;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE A &operator-=(A &a, const complex<B> &b)
	{
		a -= b.real();
		return a;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE A &operator*=(A &a, const complex<B> &b)
	{
		a *= b.real();
		return a;
	}

	template<typename A, typename B,
		typename std::enable_if<std::is_arithmetic<A>::value, int>::type = 0>
	LR_INLINE A &operator/=(A &a, const complex<B> &b)
	{
		a /= b.real();
		return a;
	}

	template<class T>
	std::ostream &operator<<(std::ostream &os, const complex<T> &val)
	{
		return os << val.str();
	}
}

namespace std
{
	// GENERATED
	template<>
	struct common_type<bool, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<char, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<unsigned char, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<int, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<unsigned int, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<long, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<unsigned long, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<long long, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<unsigned long long, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<float, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<double, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<float>, librapid::complex<double>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, bool> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, char> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, unsigned char> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, int> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, unsigned int> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, long> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, unsigned long> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, long long> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, unsigned long long> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, float> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, double> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, librapid::complex<float>> { using type = librapid::complex<double>; };
	template<>
	struct common_type<librapid::complex<double>, librapid::complex<double>> { using type = librapid::complex<double>; };
	// END GENERATED
}

#endif