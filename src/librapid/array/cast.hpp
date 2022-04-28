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
				if constexpr (internal::traits<Scalar>::PacketWidth == 8) {
					return Packet(m_toCast.scalar(index + 0),
								  m_toCast.scalar(index + 1),
								  m_toCast.scalar(index + 2),
								  m_toCast.scalar(index + 3),
								  m_toCast.scalar(index + 4),
								  m_toCast.scalar(index + 5),
								  m_toCast.scalar(index + 6),
								  m_toCast.scalar(index + 7));
				}

				if constexpr (internal::traits<Scalar>::PacketWidth == 16) {
					return Packet(m_toCast.scalar(index + 0),
								  m_toCast.scalar(index + 1),
								  m_toCast.scalar(index + 2),
								  m_toCast.scalar(index + 3),
								  m_toCast.scalar(index + 4),
								  m_toCast.scalar(index + 5),
								  m_toCast.scalar(index + 6),
								  m_toCast.scalar(index + 7),
								  m_toCast.scalar(index + 8),
								  m_toCast.scalar(index + 9),
								  m_toCast.scalar(index + 10),
								  m_toCast.scalar(index + 11),
								  m_toCast.scalar(index + 12),
								  m_toCast.scalar(index + 13),
								  m_toCast.scalar(index + 14),
								  m_toCast.scalar(index + 15));
				}

				if constexpr (internal::traits<Scalar>::PacketWidth == 32) {
					return Packet(m_toCast.scalar(index + 0),
								  m_toCast.scalar(index + 1),
								  m_toCast.scalar(index + 2),
								  m_toCast.scalar(index + 3),
								  m_toCast.scalar(index + 4),
								  m_toCast.scalar(index + 5),
								  m_toCast.scalar(index + 6),
								  m_toCast.scalar(index + 7),
								  m_toCast.scalar(index + 8),
								  m_toCast.scalar(index + 9),
								  m_toCast.scalar(index + 10),
								  m_toCast.scalar(index + 11),
								  m_toCast.scalar(index + 12),
								  m_toCast.scalar(index + 13),
								  m_toCast.scalar(index + 14),
								  m_toCast.scalar(index + 15),
								  m_toCast.scalar(index + 16),
								  m_toCast.scalar(index + 17),
								  m_toCast.scalar(index + 18),
								  m_toCast.scalar(index + 19),
								  m_toCast.scalar(index + 20),
								  m_toCast.scalar(index + 21),
								  m_toCast.scalar(index + 22),
								  m_toCast.scalar(index + 23),
								  m_toCast.scalar(index + 24),
								  m_toCast.scalar(index + 25),
								  m_toCast.scalar(index + 26),
								  m_toCast.scalar(index + 27),
								  m_toCast.scalar(index + 28),
								  m_toCast.scalar(index + 29),
								  m_toCast.scalar(index + 30),
								  m_toCast.scalar(index + 31));
				}

				if constexpr (internal::traits<Scalar>::PacketWidth == 64) {
					return Packet(m_toCast.scalar(index + 0),
								  m_toCast.scalar(index + 1),
								  m_toCast.scalar(index + 2),
								  m_toCast.scalar(index + 3),
								  m_toCast.scalar(index + 4),
								  m_toCast.scalar(index + 5),
								  m_toCast.scalar(index + 6),
								  m_toCast.scalar(index + 7),
								  m_toCast.scalar(index + 8),
								  m_toCast.scalar(index + 9),
								  m_toCast.scalar(index + 10),
								  m_toCast.scalar(index + 11),
								  m_toCast.scalar(index + 12),
								  m_toCast.scalar(index + 13),
								  m_toCast.scalar(index + 14),
								  m_toCast.scalar(index + 15),
								  m_toCast.scalar(index + 16),
								  m_toCast.scalar(index + 17),
								  m_toCast.scalar(index + 18),
								  m_toCast.scalar(index + 19),
								  m_toCast.scalar(index + 20),
								  m_toCast.scalar(index + 21),
								  m_toCast.scalar(index + 22),
								  m_toCast.scalar(index + 23),
								  m_toCast.scalar(index + 24),
								  m_toCast.scalar(index + 25),
								  m_toCast.scalar(index + 26),
								  m_toCast.scalar(index + 27),
								  m_toCast.scalar(index + 28),
								  m_toCast.scalar(index + 29),
								  m_toCast.scalar(index + 30),
								  m_toCast.scalar(index + 31),
								  m_toCast.scalar(index + 32),
								  m_toCast.scalar(index + 33),
								  m_toCast.scalar(index + 34),
								  m_toCast.scalar(index + 35),
								  m_toCast.scalar(index + 36),
								  m_toCast.scalar(index + 37),
								  m_toCast.scalar(index + 38),
								  m_toCast.scalar(index + 39),
								  m_toCast.scalar(index + 40),
								  m_toCast.scalar(index + 41),
								  m_toCast.scalar(index + 42),
								  m_toCast.scalar(index + 43),
								  m_toCast.scalar(index + 44),
								  m_toCast.scalar(index + 45),
								  m_toCast.scalar(index + 46),
								  m_toCast.scalar(index + 47),
								  m_toCast.scalar(index + 48),
								  m_toCast.scalar(index + 49),
								  m_toCast.scalar(index + 50),
								  m_toCast.scalar(index + 51),
								  m_toCast.scalar(index + 52),
								  m_toCast.scalar(index + 53),
								  m_toCast.scalar(index + 54),
								  m_toCast.scalar(index + 55),
								  m_toCast.scalar(index + 56),
								  m_toCast.scalar(index + 57),
								  m_toCast.scalar(index + 58),
								  m_toCast.scalar(index + 59),
								  m_toCast.scalar(index + 60),
								  m_toCast.scalar(index + 61),
								  m_toCast.scalar(index + 62),
								  m_toCast.scalar(index + 63));
				}

				LR_ASSERT(false, "Unknown Packet Width");
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				return Scalar(m_toCast.scalar(index));
			}

		private:
			const InputType &m_toCast;
		};
	} // namespace unary
} // namespace librapid
