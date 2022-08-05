#pragma once

#include "config.hpp"

namespace librapid {
	namespace memory {
		template<typename T, typename d>
		class ValueReference;

		template<typename T = unsigned char, typename d = device::CPU>
		class DenseStorage;
	} // namespace memory

	namespace internal {
		template<typename T>
		struct traits;
	}

	template<typename T>
	class Complex;

	template<typename T>
	T real(const Complex<T> &val);

	template<typename T>
	T imag(const Complex<T> &val);

	template<typename T, int64_t maxDims, int64_t align>
	class ExtentType;

	template<typename ArrT>
	class CommaInitializer;

	template<typename Derived, typename device>
	class ArrayBase;

	namespace unary {
		template<typename DST, typename OtherDerived>
		class Cast;
	}

	namespace binop {
		template<typename Binop, typename Derived, typename OtherDerived>
		class CWiseBinop;
	}

	namespace unop {
		template<typename Unop, typename Derived>
		class CWiseUnop;
	}

	template<typename Scalar_, typename Device_>
	class Array;

	class StrOpt {
	public:
		StrOpt() : digits(-1), base(10), scientific(false) {}

		StrOpt(int64_t digits_, int64_t base_, bool scientific_) :
				digits(digits_), base(base_), scientific(scientific_) {}

	public:
		int64_t digits;
		int64_t base;
		bool scientific;
	};

#define DEFAULT_STR_OPT StrOpt(-1, 10, false)
} // namespace librapid
