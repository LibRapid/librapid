#pragma once

#include "config.hpp"

namespace librapid {
	namespace memory {
		template<typename T, typename d>
		class ValueReference;

		template<typename T = unsigned char, typename d = device::CPU>
		class DenseStorage;
	}

	namespace internal {
		template<typename T>
		struct traits;
	}

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

	struct StrOpt {
		int64_t digits	= -1;
		int64_t base	= 10;
		bool scientific = false;
	};

#define DEFAULT_STR_OPT {-1, 10, false}
} // namespace librapid
