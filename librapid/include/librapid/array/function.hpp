#ifndef LIBRAPID_ARRAY_FUNCTION_HPP
#define LIBRAPID_ARRAY_FUNCTION_HPP

namespace librapid {
	namespace typetraits {
		// Extract allowVectorisation from the input types
		template<typename First, typename... T>
		constexpr bool checkAllowVectorisation() {
			if constexpr (sizeof...(T) == 0) {
				return TypeInfo<std::decay_t<First>>::allowVectorisation;
			} else {
				return TypeInfo<std::decay_t<First>>::allowVectorisation &&
					   checkAllowVectorisation<T...>();
			}
		}

		// template<typename First, typename... Rest>
		// struct DeviceCheckAndExtract {
		// 	using Device = typename TypeInfo<std::decay_t<First>>::Device;
		// };

		template<typename First, typename... Rest>
		constexpr auto commonDevice() {
			using FirstDevice = typename TypeInfo<std::decay_t<First>>::Device;
			if constexpr (sizeof...(Rest) == 0) {
				return FirstDevice {};
			} else {
				using RestDevice = decltype(commonDevice<Rest...>());
				if constexpr (std::is_same_v<FirstDevice, device::GPU> ||
							  std::is_same_v<RestDevice, device::GPU>) {
					return device::GPU {};
				} else {
					return device::CPU {};
				}
			}
		}

		template<typename desc, typename Functor_, typename... Args>
		struct TypeInfo<::librapid::detail::Function<desc, Functor_, Args...>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::ArrayFunction;
			using Scalar							   = decltype(std::declval<Functor_>()(
				std::declval<typename TypeInfo<std::decay_t<Args>>::Scalar>()...));
			using Device =
			  decltype(commonDevice<Args...>()); // typename DeviceCheckAndExtract<Args...>::Device;

			static constexpr bool allowVectorisation = checkAllowVectorisation<Args...>();

			static constexpr bool supportsArithmetic = TypeInfo<Scalar>::supportsArithmetic;
			static constexpr bool supportsLogical	 = TypeInfo<Scalar>::supportsLogical;
			static constexpr bool supportsBinary	 = TypeInfo<Scalar>::supportsBinary;
		};
	} // namespace typetraits

	namespace detail {
		// Descriptor is defined in "forward.hpp"

		template<
		  typename Packet, typename T,
		  typename std::enable_if_t<
			typetraits::TypeInfo<T>::type != ::librapid::detail::LibRapidType::Scalar, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packetExtractor(const T &obj,
																		 size_t index) {
			static_assert(std::is_same_v<Packet, decltype(obj.packet(index))>,
						  "Packet types do not match");
			return obj.packet(index);
		}

		template<
		  typename Packet, typename T,
		  typename std::enable_if_t<
			typetraits::TypeInfo<T>::type == ::librapid::detail::LibRapidType::Scalar, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packetExtractor(const T &obj, size_t) {
			return Packet(obj);
		}

		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type !=
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalarExtractor(const T &obj, size_t index) {
			return obj.scalar(index);
		}

		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type ==
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalarExtractor(const T &obj, size_t) {
			return obj;
		}

		template<typename First, typename... Rest>
		constexpr auto scalarTypesAreSame() {
			if constexpr (sizeof...(Rest) == 0) {
				using Scalar = typename typetraits::TypeInfo<std::decay_t<First>>::Scalar;
				return Scalar {};
			} else {
				using RestType = decltype(scalarTypesAreSame<Rest...>());
				if constexpr (std::is_same_v<
								typename typetraits::TypeInfo<std::decay_t<First>>::Scalar,
								RestType>) {
					return RestType {};
				} else {
					return std::false_type {};
				}
			}
		}

		// template<typename First, typename... Rest>
		// constexpr bool scalarTypesAreSame(const std::tuple<First, Rest...> &tup) {
		// 	constexpr auto ret = scalarTypesAreSameImpl(tup);
		// 	return !std::is_same_v<decltype(ret), std::false_type>;
		// };

		template<typename desc, typename Functor_, typename... Args>
		class Function {
		public:
			using Type		 = Function<desc, Functor_, Args...>;
			using Functor	 = Functor_;
			using Scalar	 = typename typetraits::TypeInfo<Type>::Scalar;
			using Device	 = typename typetraits::TypeInfo<Type>::Device;
			using Packet	 = typename typetraits::TypeInfo<Scalar>::Packet;
			using Descriptor = desc;
			static constexpr bool argsAreSameType =
			  !std::is_same_v<decltype(scalarTypesAreSame<Args...>()), std::false_type>;

			Function() = default;

			/// Constructs a Function from a functor and arguments.
			/// \param functor The functor to use.
			/// \param args The arguments to use.
			LIBRAPID_ALWAYS_INLINE explicit Function(Functor &&functor, Args &&...args);

			/// Constructs a Function from another function.
			/// \param other The Function to copy.
			LIBRAPID_ALWAYS_INLINE Function(const Function &other) = default;

			/// Construct a Function from a temporary function.
			/// \param other The Function to move.
			LIBRAPID_ALWAYS_INLINE Function(Function &&other) noexcept = default;

			/// Assigns a Function to this function.
			/// \param other The Function to copy.
			/// \return A reference to this Function.
			LIBRAPID_ALWAYS_INLINE Function &operator=(const Function &other) = default;

			/// Assigns a temporary Function to this Function.
			/// \param other The Function to move.
			/// \return A reference to this Function.
			LIBRAPID_ALWAYS_INLINE Function &operator=(Function &&other) noexcept = default;

			/// Return the shape of the Function's result
			/// \return The shape of the Function's result
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto shape() const;

			/// Return the arguments in the Function
			/// \return The arguments in the Function
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto &args() const;

			/// Return an evaluated Array object
			/// \return
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const;

			/// Evaluates the function at the given index, returning a Packet result.
			/// \param index The index to evaluate at.
			/// \return The result of the function (vectorized).
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packet(size_t index) const;

			/// Evaluates the function at the given index, returning a Scalar result.
			/// \param index The index to evaluate at.
			/// \return The result of the function (scalar).
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar scalar(size_t index) const;

			/// Return a string representation of the Function
			/// \param format The format to use.
			/// \return A string representation of the Function
			LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const;

		private:
			/// Implementation detail -- evaluates the function at the given index,
			/// returning a Packet result.
			/// \tparam I The index sequence.
			/// \param index The index to evaluate at.
			/// \return The result of the function (vectorized).
			template<size_t... I>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packetImpl(std::index_sequence<I...>,
																		size_t index) const;

			/// Implementation detail -- evaluates the function at the given index,
			/// returning a Scalar result.
			/// \tparam I The index sequence.
			/// \param index The index to evaluate at.
			/// \return The result of the function (scalar).
			template<size_t... I>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar scalarImpl(std::index_sequence<I...>,
																		size_t index) const;

			Functor m_functor;
			std::tuple<Args...> m_args;
		};

		template<typename desc, typename Functor, typename... Args>
		Function<desc, Functor, Args...>::Function(Functor &&functor, Args &&...args) :
				m_functor(std::forward<Functor>(functor)), m_args(std::forward<Args>(args)...) {}

		template<typename desc, typename Functor, typename... Args>
		auto Function<desc, Functor, Args...>::shape() const {
			// return std::get<0>(m_args).shape();
			return typetraits::TypeInfo<Functor>::getShape(m_args);
		}

		template<typename desc, typename Functor, typename... Args>
		auto &Function<desc, Functor, Args...>::args() const {
			return m_args;
		}

		template<typename desc, typename Functor, typename... Args>
		auto Function<desc, Functor, Args...>::eval() const {
			Array<Scalar, Device> res(shape());
			res = *this;
			return res;
		}

		template<typename desc, typename Functor, typename... Args>
		typename Function<desc, Functor, Args...>::Packet
		Function<desc, Functor, Args...>::packet(size_t index) const {
			return packetImpl(std::make_index_sequence<sizeof...(Args)>(), index);
		}

		template<typename desc, typename Functor, typename... Args>
		template<size_t... I>
		auto Function<desc, Functor, Args...>::packetImpl(std::index_sequence<I...>,
														  size_t index) const -> Packet {
			// return m_functor.packet((std::get<I>(m_args).packet(index))...);
			return m_functor.packet(packetExtractor<Packet>(std::get<I>(m_args), index)...);
		}

		template<typename desc, typename Functor, typename... Args>
		auto Function<desc, Functor, Args...>::scalar(size_t index) const -> Scalar {
			return scalarImpl(std::make_index_sequence<sizeof...(Args)>(), index);
		}

		template<typename desc, typename Functor, typename... Args>
		template<size_t... I>
		auto Function<desc, Functor, Args...>::scalarImpl(std::index_sequence<I...>,
														  size_t index) const -> Scalar {
			// return m_functor((std::get<I>(m_args).scalar(index))...);
			return m_functor(scalarExtractor(std::get<I>(m_args), index)...);
		}

		template<typename desc, typename Functor, typename... Args>
		std::string Function<desc, Functor, Args...>::str(const std::string &format) const {
			return eval().str(format);
		}
	} // namespace detail
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename desc COMMA typename Functor COMMA typename... Args,
						librapid::detail::Function<desc COMMA Functor COMMA Args...>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_FUNCTION_HPP