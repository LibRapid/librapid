#pragma once

namespace librapid {
	namespace internal {
		template<typename DST, typename OtherDerived>
		struct traits<unary::Cast<DST, OtherDerived>> {
			static constexpr bool IsScalar	= false;
			static constexpr bool IsEvaluated = true;
			using Valid						= std::true_type;
			using Type						= unary::Cast<DST, OtherDerived>;
			using Scalar					= DST;
			using Packet					= typename traits<Scalar>::Packet;
			using Device					= typename internal::traits<OtherDerived>::Device;
			using StorageType				= memory::DenseStorage<Scalar, Device>;
			static constexpr ui64 Flags = internal::flags::PythonFlags;
		};

		template<typename OtherDerived>
		struct traits<unary::Cast<bool, OtherDerived>> {
			static constexpr bool IsScalar = false;
			static constexpr bool IsEvaluated = true;
			using Valid					   = std::true_type;
			using Type					   = unary::Cast<bool, OtherDerived>;
			using Scalar				   = bool;
			using Packet				   = typename traits<Scalar>::Packet;
			using Device				   = typename internal::traits<OtherDerived>::Device;
			using StorageType			   = memory::DenseStorage<Scalar, Device>;
			static constexpr ui64 Flags =
			  internal::flags::PythonFlags | internal::flags::NoPacketOp;
		};
	} // namespace internal

	namespace unary {
		template<typename DST, typename OtherDerived>
		class Cast : public ArrayBase<Cast<DST, OtherDerived>,
									  typename internal::traits<OtherDerived>::Device> {
		public:
			using Scalar					= DST;
			using Packet					= typename internal::traits<Scalar>::Packet;
			using Device					= typename internal::traits<OtherDerived>::Device;
			using InputType					= OtherDerived;
			using InputScalar				= typename internal::traits<InputType>::Scalar;
			using Type						= Cast<DST, OtherDerived>;
			using Base						= ArrayBase<Cast<DST, OtherDerived>, Device>;
			static constexpr ui64 Flags = internal::traits<Type>::Flags;

			Cast() = delete;

			Cast(const InputType &toCast) : Base(toCast.extent()), m_toCast(toCast) {}

			Cast(const Type &caster) : Base(caster.extent()), m_toCast(caster.m_toCast) {}

			Cast &operator=(const Type &caster) {
				if (this == &caster) return *this;
				Base::m_extent = caster.m_extent;
				Base::m_data   = caster.m_data;
				m_toCast	   = caster.m_toCast;
				return *this;
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				Array<Scalar, Device> res(Base::extent());
				res.assign(*this);
				return res;
			}

			Packet packet(i64 index) const {
				// Quick return if possible
				if constexpr (std::is_same_v<Scalar, InputScalar>) return m_toCast.packet(index);
				static Scalar buffer[Packet::size()];
				for (i64 i = 0; i < Packet::size(); ++i)
					buffer[i] = internal::traits<InputScalar>::template cast<Scalar>(
					  m_toCast.scalar(index + i));
				return Packet(&(buffer[0]));
			}

			Scalar scalar(i64 index) const {
				return internal::traits<InputScalar>::template cast<Scalar>(m_toCast.scalar(index));
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, i64 &index) const {
				return fmt::format("(({}) ({}))", internal::traits<Scalar>::Name, 5);
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							i64 stripWidth = -1, i64 beforePoint = -1,
							i64 afterPoint = -1, i64 depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

		private:
			InputType m_toCast;
		};
	} // namespace unary
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename DST, typename OtherDerived>
struct fmt::formatter<librapid::unary::Cast<DST, OtherDerived>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::unary::Cast<DST, OtherDerived> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API
