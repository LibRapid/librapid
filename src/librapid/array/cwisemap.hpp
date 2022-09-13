#pragma once

#include <utility>

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/kernelFormat.hpp"

namespace librapid {
	namespace mapping {
		template<typename First>
		constexpr bool allSameDevice() {
			return true;
		}

		template<typename First, typename Second>
		constexpr bool allSameDevice() {
			return std::is_same_v<typename internal::traits<First>::Device,
								  typename internal::traits<Second>::Device>;
		}

		template<typename First, typename Second, typename... Rest,
				 typename std::enable_if_t<(sizeof...(Rest) > 0)> = 0>
		constexpr bool allSameDevice() {
			return allSameDevice<First, Second>() && allSameDevice<Second, Rest...>();
		}

		template<typename T, typename = int>
		struct HasFlags : std::false_type {};

		template<typename T>
		struct HasFlags<T, decltype((void)T::Flags, 0)> : std::true_type {};

		template<typename Map>
		constexpr uint64_t extractFlags() {
			if constexpr (HasFlags<Map>::value) {
				return Map::Flags;
			} else {
				return 0;
			}
		}

		template<typename First>
		LR_FORCE_INLINE auto extractAndCheckExtent(const First &first) {
			if constexpr (internal::traits<First>::IsScalar) {
				return first;
			} else {
				return first.extent();
			}
		}

		template<typename First, typename... Rest>
		LR_FORCE_INLINE auto extractAndCheckExtent(const First &first, const Rest &...rest) {
			if constexpr (internal::traits<First>::IsScalar) {
				return extractAndCheckExtent(rest...);
			} else {
				if constexpr ((internal::traits<Rest>::IsScalar && ...)) {
					return first.extent();
				} else {
					LR_ASSERT(first.extent() == extractAndCheckExtent(rest...),
							  "All arrays must have the same extent");
					return first.extent();
				}
			}
		}

		template<typename Scalar, typename T>
		LR_FORCE_INLINE auto extractPacket(const T &val, int64_t index) {
			if constexpr (internal::traits<T>::IsScalar) {
				using Packet = typename internal::traits<Scalar>::Packet;
				return Packet(static_cast<Scalar>(val));
			} else {
				return val.packet(index);
			}
		}

		template<typename Scalar, typename T>
		LR_FORCE_INLINE auto extractScalar(const T &val, int64_t index) {
			if constexpr (internal::traits<T>::IsScalar) {
				return static_cast<Scalar>(val);
			} else {
				return static_cast<Scalar>(val.scalar(index));
			}
		}

		// Count the number of unique mapping objects ever created
		static inline std::atomic<uint64_t> lambdaMapCount = 0;
	} // namespace mapping

	namespace internal {
		template<bool allowPacket, typename Map, typename... DerivedTypes>
		struct traits<mapping::CWiseMap<allowPacket, Map, DerivedTypes...>> {
			static_assert(mapping::allSameDevice<DerivedTypes...>(),
						  "All arrays must be on the same device");

			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = false;
			using Valid						  = std::true_type;
			using Type	 = mapping::CWiseMap<allowPacket, Map, DerivedTypes...>;
			using Scalar = typename std::common_type_t<typename traits<DerivedTypes>::Scalar...>;
			using BaseScalar = typename traits<Scalar>::BaseScalar;
			using Packet = typename std::conditional_t<allowPacket, typename traits<Scalar>::Packet,
													   std::false_type>;
			using Device =
			  typename memory::PromoteDeviceMulti<typename traits<DerivedTypes>::Device...>;
			using StorageType				= memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags = flags::CustomFunctionGen |
											  mapping::extractFlags<Map>() |
											  (allowPacket ? 0 : flags::NoPacketOp);
		};
	} // namespace internal

	namespace mapping {
		template<bool allowPacket, typename Map, typename... DerivedTypes>
		class CWiseMap : public ArrayBase<CWiseMap<allowPacket, Map, DerivedTypes...>,
										  typename internal::traits<
											CWiseMap<allowPacket, Map, DerivedTypes...>>::Device> {
		public:
			using Operation					= Map;
			using Scalar					= typename internal::traits<CWiseMap>::Scalar;
			using Packet					= typename internal::traits<Scalar>::Packet;
			using Device					= typename internal::traits<CWiseMap>::Device;
			using Type						= CWiseMap<allowPacket, Map, DerivedTypes...>;
			using Base						= ArrayBase<Type, Device>;
			static constexpr uint64_t Flags = internal::traits<Type>::Flags;

			CWiseMap() = delete;

			CWiseMap(const Map &map, const DerivedTypes &...args) :
					Base(extractAndCheckExtent(args...), 0), m_operation(map),
					m_operands(std::make_tuple(args...)),
					m_name(fmt::format("implMapFunction_{}", lambdaMapCount)) {}

			CWiseMap(const CWiseMap &op) :
					Base(op.extent(), 0), m_operation(op.m_operation), m_operands(op.m_operands),
					m_name(op.m_name) {}

			CWiseMap &operator=(const CWiseMap &op) = delete;

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
				Array<Scalar, Device> res(Base::extent());
				res.assign(*this);
				return res;
			}

			LR_FORCE_INLINE auto packet(int64_t index) const {
				return std::apply(m_operation,
								  std::apply(
									[index](auto &&...args) {
										return std::make_tuple(
										  extractPacket<Scalar>(args, index)...);
									},
									m_operands));
			}

			LR_FORCE_INLINE auto scalar(int64_t index) const {
				return std::apply(m_operation,
								  std::apply(
									[index](auto &&...args) {
										return std::make_tuple(
										  extractScalar<Scalar>(args, index)...);
									},
									m_operands));
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

			LR_NODISCARD("") std::string genCustomFunction() const {
				std::string types;
				std::string args;
				std::vector<std::string> argNames = m_operation.args();
				for (int64_t i = 0; i < argNames.size(); ++i) {
					std::string type = fmt::format("T_{}", i);
					types += fmt::format("typename {}", type);
					args += fmt::format("{} {}", type, argNames[i]);
					if (i + 1 < argNames.size()) {
						args += ", ";
						types += ", ";
					}
				}
				return fmt::format(R"V0G0N(
template<{0}>
__forceinline__  __device__ {1} {2}({3}) {{
	{3}
}}
								   )V0G0N",
								   types,						   // Templates
								   internal::traits<Scalar>::Name, // Return type
								   m_name,						   // Function name
								   args,						   // Arguments
								   m_operation.kernel());		   // Kernel
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				std::vector<std::string> strings;
				extractTupleInfo<0>(strings, vec, index, m_operands);

				std::string res("customFunctionImpl(");
				for (int64_t i = 0; i < strings.size(); ++i) {
					res += strings[i];
					if (i + 1 < strings.size()) { res += ", "; }
				}

				return res + ")";
			}

			LR_NODISCARD("") std::string name() const { return m_name; }

		private:
			template<int64_t TupleIndex, typename T, typename... Pack>
			LR_FORCE_INLINE void extractTupleInfo(std::vector<std::string> &strings,
												  std::vector<T> &vec, int64_t &index,
												  const std::tuple<Pack...> &vals) const {
				if constexpr (TupleIndex < sizeof...(Pack)) {
					if constexpr (internal::traits<typename std::tuple_element<
									TupleIndex,
									std::tuple<Pack...>>::type>::IsScalar) {
						strings.emplace_back(detail::kernelFormat(std::get<0>(vals)));
					} else {
						strings.emplace_back(std::get<0>(vals).genKernel(vec, index));
					}

					if constexpr (TupleIndex + 1 < sizeof...(Pack)) {
						extractTupleInfo<TupleIndex + 1>(strings, vec, index, vals);
					}
				}
			}

		private:
			Map m_operation;
			std::tuple<DerivedTypes...> m_operands;
			std::string m_name;
		};
	} // namespace mapping

	// Arbitrary mapping of functions to arrays
	template<bool allowPacket = true, bool forceTemporary = false, typename Map,
			 typename... DerivedTypes>
	LR_NODISCARD("")
	auto map(const Map &map, DerivedTypes... args) {
		using Scalar =
		  typename std::common_type_t<typename internal::traits<DerivedTypes>::Scalar...>;
		using BaseScalar				   = typename internal::traits<Scalar>::BaseScalar;
		using RetType					   = mapping::CWiseMap<allowPacket, Map, DerivedTypes...>;
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;

		static_assert(!(Required & ~(Flags & Required)),
					  "Scalar type is incompatible with Functor");

		if constexpr (!forceTemporary && // If we REQUIRE a temporary value, don't evaluate it
					  ((bool)(Flags & internal::flags::RequireEval)))
			return RetType(map, args...).eval();
		else
			return RetType(map, args...);
	}
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<bool allowPacket, typename Map, typename... DerivedTypes>
struct fmt::formatter<librapid::mapping::CWiseMap<allowPacket, Map, DerivedTypes...>> {
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
	auto format(const librapid::mapping::CWiseMap<allowPacket, Map, DerivedTypes...> &map,
				FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), map.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API