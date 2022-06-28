#pragma once

#include "../../../internal/config.hpp"
#include "../traits.hpp"

namespace librapid::internal {
	//------- 16bit Floating Point --------------------------------------------
	template<>
	struct traits<extended::float16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = extended::float16_t;
		using BaseScalar					 = extended::float16_t;
		using StorageType					 = memory::DenseStorage<extended::float16_t>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "__half";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};
} // namespace librapid::internal