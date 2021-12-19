#ifndef LIBRAPID_CUSTOM_COMPLEX
#define LIBRAPID_CUSTOM_COMPLEX

#include <librapid/stringmethods/format_number.hpp>
#include <complex>

namespace librapid {
    template<typename T>
    class Complex;

    template<typename T>
    std::string format_number(const T &val);
    template<typename T>
    std::string format_number(const Complex<T> &val);

	template<class T>
	class Complex {
	public:
		Complex(const T &real_val = T(), const T &imag_val = T())
				: m_real(real_val), m_imag(imag_val) {}

		Complex &operator=(const T &val) {
			m_real = val;
			m_imag = 0;
			return *this;
		}

		template<class V>
		Complex(const Complex<V> &other)
				: Complex(static_cast<T>(other.real()), static_cast<T>(other.imag())) {}

		template<class V>
		Complex &operator=(const Complex<V> &other) {
			m_real = static_cast<T>(other.real());
			m_imag = static_cast<T>(other.imag());
			return *this;
		}

		Complex copy() const {
			return Complex<T>(m_real, m_imag);
		}

		Complex operator-() const {
			return Complex<T>(-m_real, -m_imag);
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex operator+(const V &other) const {
			return Complex<T>(m_real + other, m_imag);
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex operator-(const V &other) const {
			return Complex<T>(m_real - other, m_imag);
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex operator*(const V &other) const {
			return Complex<T>(m_real * other, m_imag * other);
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex operator/(const V &other) const {
			return Complex<T>(m_real / other, m_imag / other);
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex &operator+=(const V &other) {
			m_real += other;
			return *this;
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex &operator-=(const V &other) {
			m_real -= other;
			return *this;
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex &operator*=(const V &other) {
			m_real *= other;
			m_imag *= other;
			return *this;
		}

		template<typename V, typename std::enable_if<std::is_scalar<V>::value, int>::type = 0>
		Complex &operator/=(const V &other) {
			m_real /= other;
			m_imag /= other;
			return *this;
		}

		template<typename V>
		Complex operator+(const Complex<V> &other) const {
			return Complex(m_real + other.real(),
						   m_imag + other.imag());
		}

		template<typename V>
		Complex operator-(const Complex<V> &other) const {
			return Complex(m_real - other.real(),
						   m_imag - other.imag());
		}

		template<typename V>
		Complex operator*(const Complex<V> &other) const {
			return Complex((m_real * other.real()) - (m_imag * other.imag()),
						   (m_real * other.imag()) + (m_imag * other.real()));
		}

		template<typename V>
		Complex operator/(const Complex<V> &other) const {
			return Complex((m_real * other.real()) + (m_imag * other.imag()) /
													 ((other.real() * other.real()) + (other.imag() * other.imag())),
						   (m_real * other.real()) - (m_imag * other.imag()) /
													 ((other.real() * other.real()) + (other.imag() * other.imag())));
		}

		template<typename V>
		Complex &operator+=(const Complex<V> &other) {
			m_real = m_real + other.real();
			m_imag = m_imag + other.imag();
			return *this;
		}

		template<typename V>
		Complex &operator-=(const Complex<V> &other) {
			m_real = m_real - other.real();
			m_imag = m_imag - other.imag();
			return *this;
		}

		template<typename V>
		Complex &operator*=(const Complex<V> &other) {
			m_real = (m_real * other.real()) - (m_imag * other.imag());
			m_imag = (m_real * other.imag()) + (imag() * other.real());
			return *this;
		}

		template<typename V>
		Complex &operator/=(const Complex<V> &other) {
			m_real = (m_real * other.real()) + (m_imag * other.imag()) /
											   ((other.real() * other.real()) + (other.imag() * other.imag()));
			m_imag = (m_real * other.real()) - (m_imag * other.imag()) /
											   ((other.real() * other.real()) + (other.imag() * other.imag()));
			return *this;
		}

		template<typename V>
		bool operator==(const Complex<V> &other) const {
			return m_real == other.real() && m_imag == other.imag();
		}

		template<typename V>
		bool operator!=(const Complex<V> &other) const {
			return !(*this == other);
		}

		template<typename V>
		bool operator==(const V &other) const {
			return m_real == other && m_imag == 0;
		}

		template<typename V>
		bool operator!=(const V &other) const {
			return !(*this == other);
		}

		T mag() const {
			return std::sqrt(m_real * m_real + m_imag * m_imag);
		}

		T angle() const {
			return std::atan2(m_real, m_imag);
		}

		Complex<T> log() const {
			return Complex<T>(std::log(mag()), angle());
		}

		Complex<T> conjugate() const {
			return Complex<T>(m_real, -m_imag);
		}

		Complex<T> reciprocal() const {
			return Complex<T>((m_real) / (m_real * m_real + m_imag * m_imag),
							  -(m_imag) / (m_real * m_real + m_imag * m_imag));
		}

		const T &real() const {
			return m_real;
		}

		T &real() {
			return m_real;
		}

		const T &imag() const {
			return m_imag;
		}

		T &imag() {
			return m_imag;
		}

		explicit operator std::string() const {
			return str();
		}

		template<typename V>
		operator V() const {
			return m_real;
		}

		template<typename V>
		explicit operator std::complex<V>() const {
			return std::complex<V>(m_real, m_imag);
		}

		[[nodiscard]] std::string str() const {
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

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	Complex<B> operator+(const A &a, const Complex<B> &b) {
		return Complex<B>(a) + b;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	Complex<B> operator-(const A &a, const Complex<B> &b) {
		return Complex<B>(a) - b;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	Complex<B> operator*(const A &a, const Complex<B> &b) {
		return Complex<B>(a) * b;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	Complex<B> operator/(const A &a, const Complex<B> &b) {
		return Complex<B>(a) / b;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	A &operator+=(A &a, const Complex<B> &b) {
		a += b.real();
		return a;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	A &operator-=(A &a, const Complex<B> &b) {
		a -= b.real();
		return a;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	A &operator*=(A &a, const Complex<B> &b) {
		a *= b.real();
		return a;
	}

	template<typename A, typename B, typename std::enable_if<std::is_scalar<A>::value, int>::type = 0>
	A &operator/=(A &a, const Complex<B> &b) {
		a /= b.real();
		return a;
	}

	template<class T>
	std::ostream &operator<<(std::ostream &os, const Complex<T> &val) {
		return os << val.str();
	}

	template<typename T>
	std::string format_number(const Complex<T> &val) {
		std::stringstream stream;
		stream.precision(10);
		stream << val;
		return stream.str();
	}
}

#endif