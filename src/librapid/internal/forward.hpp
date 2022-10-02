#pragma once

namespace librapid {
	namespace memory {
		template<typename T, typename d>
		class ValueReference;

		template<typename T, typename d>
		class DenseStorage;
	} // namespace memory

	namespace internal {
		template<typename T>
		struct traits;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	class VecImpl;

	template<typename T>
	class Complex;

	template<typename T>
	T real(const Complex<T> &val);

	template<typename T>
	T imag(const Complex<T> &val);

	template<typename T, i32 maxDims, i32 align>
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

	namespace mapping {
		template<bool allowPacket, typename Map, typename... DerivedTypes>
		class CWiseMap;
	}

	template<typename Scalar_, typename Device_>
	class Array;

	class StrOpt {
	public:
		StrOpt() : digits(-1), base(10), scientific(false) {}

		StrOpt(i32 digits_, i8 base_, bool scientific_) :
				digits(digits_), base(base_), scientific(scientific_) {}

	public:
		i32 digits;
		i8 base;
		bool scientific;
	};

#define DEFAULT_STR_OPT StrOpt(-1, 10, false)
} // namespace librapid
