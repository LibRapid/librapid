#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/kernelFormat.hpp"
#include "arrayBase.hpp"

namespace librapid {
	namespace internal {
		template<typename Binop, typename LHS, typename RHS>
		struct traits<binop::CWiseBinop<Binop, LHS, RHS>> {
			static constexpr bool IsScalar = false;
			using Valid					   = std::true_type;
			using Type					   = binop::CWiseBinop<Binop, LHS, RHS>;
			using Scalar				   = typename Binop::RetType;
			using BaseScalar			   = typename traits<Scalar>::BaseScalar;
			using Packet				   = typename traits<Scalar>::Packet;
			using DeviceLHS				   = typename traits<LHS>::Device;
			using DeviceRHS				   = typename traits<RHS>::Device;
			using Device	  = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
			using StorageType = memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags =
			  Binop::Flags | traits<LHS>::Flags | traits<RHS>::Flags;
		};
	} // namespace internal

	namespace binop {
		template<typename Binop, typename LHS, typename RHS>
		class CWiseBinop
				: public ArrayBase<CWiseBinop<Binop, LHS, RHS>,
								   typename internal::PropagateDeviceType<LHS, RHS>::Device> {
		public:
			using Operation = Binop;
			using Scalar	= typename Binop::RetType;
			using Packet	= typename internal::traits<Scalar>::Packet;
			using LeftType	= typename internal::StripQualifiers<LHS>;
			using RightType = typename internal::StripQualifiers<RHS>;
			using DeviceLHS = typename internal::traits<LHS>::Device;
			using DeviceRHS = typename internal::traits<RHS>::Device;
			using Device	= typename memory::PromoteDevice<DeviceRHS, DeviceLHS>::type;
			using Type		= CWiseBinop<Binop, LHS, RHS>;
			using Base		= ArrayBase<Type, Device>;
			static constexpr bool LhsIsScalar = internal::traits<LeftType>::IsScalar;
			static constexpr bool RhsIsScalar = internal::traits<RightType>::IsScalar;
			static constexpr uint64_t Flags	  = internal::traits<Type>::Flags;

			CWiseBinop() = delete;

			template<typename... Args>
			CWiseBinop(const LeftType &lhs, const RightType &rhs, Args... opArgs) :
					Base(
					  [&]() {
						  if constexpr (LhsIsScalar)
							  return rhs.extent();
						  else
							  return lhs.extent();
					  }(),
					  0),
					m_lhs(lhs), m_rhs(rhs), m_operation(opArgs...) {}

			CWiseBinop(const Type &op) :
					Base(op.extent(), 0), m_lhs(op.m_lhs), m_rhs(op.m_rhs),
					m_operation(op.m_operation) {}

			CWiseBinop &operator=(const Type &op) {
				if (this == &op) return *this;

				Base::m_extent = op.m_extent;

				m_lhs		= op.m_lhs;
				m_rhs		= op.m_rhs;
				m_operation = op.m_operation;

				return *this;
			}

			LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) const {
				LR_WARN_ONCE(
				  "Calling operator[] on a lazy-evaluation object forces evaluation every time. "
				  "Consider using operator() instead");

				auto res = eval();
				return res[index];
			}

			template<typename... T>
			LR_NODISCARD("")
			auto operator()(T... indices) const {
				LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
							sizeof...(T) == Base::extent().dims(),
						  "Array with {0} dimensions requires {0} access indices. Received {1}",
						  Base::extent().dims(),
						  sizeof...(indices));

				int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
				return scalar(index);
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				ExtentType<int64_t, 32> resExtent;
				if constexpr (LhsIsScalar && RhsIsScalar) {
					LR_ASSERT(false, "This should never happen");
				} else if constexpr (LhsIsScalar && !RhsIsScalar) {
					resExtent = m_operation.genExtent(m_rhs.extent());
				} else if constexpr (!LhsIsScalar && RhsIsScalar) {
					resExtent = m_operation.genExtent(m_lhs.extent());
				} else {
					resExtent = m_operation.genExtent(m_lhs.extent(), m_rhs.extent());
				}

				Array<Scalar, Device> res(resExtent);

				if constexpr ((bool)(Flags & internal::flags::HasCustomEval)) {
					m_operation.customEval(m_lhs, m_rhs, res);
					return res;
				}

				res.assign(*this);
				return res;
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				if constexpr (LhsIsScalar && RhsIsScalar)
					return m_operation.packetOp(m_lhs, m_rhs);
				else if constexpr (LhsIsScalar && !RhsIsScalar)
					return m_operation.packetOp(m_lhs, m_rhs.packet(index));
				else if constexpr (!LhsIsScalar && RhsIsScalar)
					return m_operation.packetOp(m_lhs.packet(index), m_rhs);
				else
					return m_operation.packetOp(m_lhs.packet(index), m_rhs.packet(index));
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				if constexpr (LhsIsScalar && RhsIsScalar)
					return m_operation.scalarOp(m_lhs, m_rhs);
				else if constexpr (LhsIsScalar && !RhsIsScalar)
					return m_operation.scalarOp(m_lhs, m_rhs.scalar(index));
				else if constexpr (!LhsIsScalar && RhsIsScalar)
					return m_operation.scalarOp(m_lhs.scalar(index), m_rhs);
				else
					return m_operation.scalarOp(m_lhs.scalar(index), m_rhs.scalar(index));
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				std::string leftKernel, rightKernel;

				if constexpr (LhsIsScalar && RhsIsScalar) {
					leftKernel	= detail::kernelFormat(m_lhs);
					rightKernel = detail::kernelFormat(m_rhs);
				} else if constexpr (LhsIsScalar && !RhsIsScalar) {
					leftKernel	= detail::kernelFormat(m_lhs);
					rightKernel = m_rhs.genKernel(vec, index);
				} else if constexpr (!LhsIsScalar && RhsIsScalar) {
					leftKernel	= m_lhs.genKernel(vec, index);
					rightKernel = detail::kernelFormat(m_rhs);
				} else {
					leftKernel	= m_lhs.genKernel(vec, index);
					rightKernel = m_rhs.genKernel(vec, index);
				}

				std::string op = m_operation.genKernel();
				return fmt::format("({} {} {})", leftKernel, op, rightKernel);
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

		private:
			LeftType m_lhs;
			RightType m_rhs;
			Binop m_operation {};
		};
	} // namespace binop
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Binop, typename LHS, typename RHS>
struct fmt::formatter<librapid::binop::CWiseBinop<Binop, LHS, RHS>> {
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
	auto format(const librapid::binop::CWiseBinop<Binop, LHS, RHS> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API