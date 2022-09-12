#pragma once

#include <utility>

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/kernelFormat.hpp"
#include "arrayBase.hpp"

namespace librapid {
	namespace mapping {
		template<typename First>
		constexpr bool allSameDevice() {
			return true;
		}

		template<typename First, typename Second>
		constexpr bool allSameDevice() {
			return std::is_same_v<First, Second>;
		}

		template<typename First, typename Second, typename... Rest,
				 typename std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
		constexpr bool allSameDevice() {
			return std::is_same_v<First, Second> && allSameDevice<Rest...>();
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

		template<typename T>
		LR_FORCE_INLINE auto extractPacket(const T &val, int64_t index) {
			if constexpr (internal::traits<T>::IsScalar) {
				using Packet = typename internal::traits<T>::Packet;
				return Packet(val);
			} else {
				return val.packet(index);
			}
		}

		template<typename T>
		LR_FORCE_INLINE auto extractScalar(const T &val, int64_t index) {
			if constexpr (internal::traits<T>::IsScalar) {
				return val;
			} else {
				return val.scalar(index);
			}
		}
	} // namespace mapping

	namespace internal {
		template<typename Map, typename... DerivedTypes>
		struct traits<mapping::CWiseMap<Map, DerivedTypes...>> {
			static_assert(mapping::allSameDevice<DerivedTypes...>(),
						  "All arrays must be on the same device");

			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = false;
			using Valid						  = std::true_type;
			using Type						  = mapping::CWiseMap<Map, DerivedTypes...>;
			using Scalar = typename std::common_type_t<typename traits<DerivedTypes>::Scalar...>;
			using BaseScalar = typename traits<Scalar>::BaseScalar;
			using Packet	 = typename traits<Scalar>::Packet;
			using Device = typename std::common_type_t<typename traits<DerivedTypes>::Device...>;
			using StorageType = memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags =
			  mapping::extractFlags<Map>() | (traits<DerivedTypes>::Flags | ...);
		};
	} // namespace internal

	namespace mapping {
		// Utility Mapping Type

		static inline std::atomic<uint64_t> mapCounter = 0;

		template<typename Map>
		class MapFunction {
		public:
			MapFunction() = default;

			MapFunction(const Map &map, std::vector<std::string> argNames = {},
						const std::string &kernel = "") :
					m_map(map),
					m_argNames(std::move(argNames)), m_kernel(kernel) {
				m_name = "jitMappingKernel_" + ::librapid::str((uint64_t)mapCounter++);
			}

			MapFunction(const MapFunction &other) {
				m_argNames = other.m_argNames;
				m_name	   = other.m_name;
				m_kernel   = other.m_kernel;
			};

			MapFunction &operator=(const MapFunction &other) {
				if (this == &other) return *this;
				m_argNames = other.m_argNames;
				m_name	   = other.m_name;
				m_kernel   = other.m_kernel;
				return *this;
			};

			template<typename... DerivedTypes>
			LR_FORCE_INLINE auto operator()(const DerivedTypes &...args) const {
				return m_map(args...);
			}

			const auto &name() const { return m_name; }
			const auto &argNames() const { return m_argNames; }

			LR_NODISCARD("") std::string genJitLambda() const {
				std::string args;
				for (const auto &arg : m_argNames) {
					args += "auto " + arg;
					if (arg != m_argNames.back()) { args += ", "; }
				}

				return fmt::format("[]({}) {{ {} }}", args, m_kernel);
			}

			// private:
		public:
			Map m_map;
			std::string m_name;
			std::vector<std::string> m_argNames;
			std::string m_kernel;
		};

		template<typename Map, typename... DerivedTypes>
		class CWiseMap : public ArrayBase<
						   CWiseMap<Map, DerivedTypes...>,
						   typename internal::traits<CWiseMap<Map, DerivedTypes...>>::Device> {
		public:
			using Operation					= Map;
			using Scalar					= typename internal::traits<CWiseMap>::Scalar;
			using Packet					= typename internal::traits<Scalar>::Packet;
			using Device					= typename internal::traits<CWiseMap>::Device;
			using Type						= CWiseMap<Map, DerivedTypes...>;
			using Base						= ArrayBase<Type, Device>;
			static constexpr uint64_t Flags = internal::traits<Type>::Flags;

			CWiseMap() = delete;

			CWiseMap(const Map &map, const DerivedTypes &...args) :
					Base(extractAndCheckExtent(args...), 0), m_operation(map),
					m_operands(std::make_tuple(args...)) {
			}

			// CWiseMap(const MapFunction<Map> &map, const DerivedTypes &...args) :
			// 		Base(extractAndCheckExtent(args...), 0), m_operation(map),
			// 		m_operands(std::make_tuple(args...)) {
			// 	fmt::print("INFORMATION HEHEHE: {}\n", map.m_kernel);
			// 	fmt::print("INFORMATION TURD FUCKER: {}\n", typeid(map).name());
			// }

			CWiseMap(const CWiseMap &op) = default;

			CWiseMap &operator=(const CWiseMap &op) = default;

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
										return std::make_tuple(extractPacket(args, index)...);
									},
									m_operands));
			}

			LR_FORCE_INLINE auto scalar(int64_t index) const {
				return std::apply(m_operation,
								  std::apply(
									[index](auto &&...args) {
										return std::make_tuple(extractScalar(args, index)...);
									},
									m_operands));
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				std::string lambda = m_operation.genJitLambda();
				fmt::print("KERNEL: {}\n", m_operation.m_kernel);
				fmt::print("LAMBDA: {}\n", lambda);
				std::string operation = lambda + "(5, 10)";
				return operation;
			}

		private:
			MapFunction<Map> m_operation;
			std::tuple<DerivedTypes...> m_operands;
		};
	} // namespace mapping
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Map, typename... DerivedTypes>
struct fmt::formatter<librapid::mapping::CWiseMap<Map, DerivedTypes...>> {
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
	auto format(const librapid::mapping::CWiseMap<Map, DerivedTypes...> &map, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), map.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API