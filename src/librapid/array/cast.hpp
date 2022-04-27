#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "arrayBase.hpp"

namespace librapid {
	namespace internal {
		template<typename DST, typename OtherDerived>
		struct traits<unary::Cast<DST, OtherDerived>> {
			using Scalar				= DST;
			using Packet				= typename traits<Scalar>::Packet;
			using Device				= typename internal::traits<OtherDerived>::Device;
			using StorageType			= memory::DenseStorage<Scalar, Device>;
			static const uint64_t Flags = 0;
		};
	} // namespace internal

	namespace unary {
		template<typename DST, typename OtherDerived>
		class Cast : public ArrayBase<Cast<DST, OtherDerived>,
									  typename internal::traits<OtherDerived>::Device> {
		public:
			using Scalar	  = DST;
			using Packet	  = typename internal::traits<Scalar>::Packet;
			using Device	  = typename internal::traits<OtherDerived>::Device;
			using InputType	  = OtherDerived;
			using InputScalar = typename internal::traits<InputType>::Scalar;
			using Type		  = Cast<DST, OtherDerived>;
			using Base		  = ArrayBase<Cast<DST, OtherDerived>, Device>;

			Cast() = delete;

			Cast(const InputType &toCast) : Base(toCast.extent()), m_toCast(toCast) {}

			Cast(const Type &caster) : Base(caster.extent()), m_toCast(caster) {}

			Cast &operator=(const Type &caster) {
				if (this == &caster) return *this;
				Base::m_extent = caster.m_extent;
				Base::m_data   = caster.m_data;
				m_toCast	   = caster.m_toCast;
				return *this;
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				Packet res;
				int64_t packetWidthRet = internal::traits<Scalar>::PacketWidth;
				int64_t packetWidthSrc = internal::traits<InputScalar>::PacketWidth;

				static InputScalar bufferSrc[64]; // TODO: Variable 64 (max buffer width)
				static Scalar bufferDst[64];

				int64_t copied = 0;

				// Load in a single operation
				while (copied < packetWidthRet) {
					auto tmp = m_toCast.packet(index + copied);
					tmp.store(&(bufferSrc[copied]));
					copied += packetWidthSrc;
				}

				for (int64_t i = 0; i < packetWidthRet; ++i) bufferDst[i] = bufferSrc[i];

				res.load(&(bufferDst[0]));

				return res;
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				return Scalar(m_toCast.scalar(index));
			}

		private:
			const InputType &m_toCast;
		};
	} // namespace unary
} // namespace librapid
