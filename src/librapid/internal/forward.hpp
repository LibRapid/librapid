#pragma once

#include "config.hpp"

namespace librapid {
	namespace memory {
		template<typename T = u_char, typename d = device::CPU>
		class DenseStorage;
	}

	namespace internal {
		template<typename T>
		struct traits;
	}

	template<typename T = int64_t, int64_t maxDims = 32>
	class Extent;

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

	template<typename Scalar_, typename Device_>
	class Array;
} // namespace librapid
