#pragma once

#include "../../internal/config.hpp"
#include "../../internal/forward.hpp"
#include "../traits.hpp"

namespace librapid { namespace functors {
	template<typename Derived, typename OtherDerived, bool evalBeforeAssign>
	struct AssignSelector;

	template<typename Derived, typename OtherDerived>
	struct AssignSelector<Derived, OtherDerived, false> {
		static Derived &run(Derived &left, const OtherDerived &right) {
			return left.assignLazy(right);
		}
	};

	template<typename Derived, typename OtherDerived>
	struct AssignOp {
		LR_FORCE_INLINE static void run(Derived &dst, const OtherDerived &src) {
			using Scalar = typename internal::traits<Derived>::Scalar;
			using Packet = typename internal::traits<Scalar>::Packet;

			int64_t packetWidth = internal::traits<Scalar>::PacketWidth;
			int64_t len			= dst.extent().size();
			int64_t alignedLen =
			  len -
			  (len % packetWidth); // len - (len & (packetWidth - 1)); // len - (len % packetWidth)

			// Use the entire packet width where possible
			if (numThreads < 2 || len < 500) {
				for (int64_t i = 0; i < alignedLen - packetWidth; i += packetWidth) {
					dst.loadFrom(i, src);
				}
			} else {
				// Multi-threaded approach
#pragma omp parallel for shared(dst, src, alignedLen, packetWidth) default(none) num_threads(numThreads)
				for (int64_t i = 0; i < alignedLen - packetWidth; i += packetWidth) {
					dst.loadFrom(i, src);
				}
			}

			// Ensure the remaining values are filled
			for (int64_t i = alignedLen - packetWidth; i < len; ++i) { dst.loadFromScalar(i, src); }
		}
	};
}} // namespace librapid::functors