#ifndef LIBRAPID_ARRAY_FUNCTION_HPP
#define LIBRAPID_ARRAY_FUNCTION_HPP

namespace librapid {
	namespace typetraits {
		template<typename First, typename... Rest>
		struct DeviceCheckAndExtract {
			using Device = typename TypeInfo<std::decay_t<First>>::Device;
		};

		template<typename desc, typename Functor_, typename... Args>
		struct TypeInfo<::librapid::detail::Function<desc, Functor_, Args...>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = decltype(std::declval<Functor_>()(
			  std::declval<typename TypeInfo<std::decay_t<Args>>::Scalar>()...));
			using Device = typename typetraits::DeviceCheckAndExtract<Args...>::Device;
			static constexpr bool supportsArithmetic = TypeInfo<Scalar>::supportsArithmetic;
			static constexpr bool supportsLogical	 = TypeInfo<Scalar>::supportsLogical;
			static constexpr bool supportsBinary	 = TypeInfo<Scalar>::supportsBinary;
		};
	} // namespace typetraits

	namespace detail {
		// Descriptor is defined in "forward.hpp"

		template<typename desc, typename Functor_, typename... Args>
		class Function {
		public:
			using Type	  = Function<desc, Functor_, Args...>;
			using Functor = Functor_;
			using Scalar  = typename typetraits::TypeInfo<Type>::Scalar;
			using Device  = typename typetraits::TypeInfo<Type>::Device;
			using Packet  = typename typetraits::TypeInfo<Scalar>::Packet;

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
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto &shape() const;

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
		auto &Function<desc, Functor, Args...>::shape() const {
			return std::get<0>(m_args).shape();
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
		typename Function<desc, Functor, Args...>::Packet
		Function<desc, Functor, Args...>::packetImpl(std::index_sequence<I...>,
													 size_t index) const {
			return m_functor.packet((std::get<I>(m_args).packet(index))...);
		}

		template<typename desc, typename Functor, typename... Args>
		auto Function<desc, Functor, Args...>::scalar(size_t index) const -> Scalar {
			return scalarImpl(std::make_index_sequence<sizeof...(Args)>(), index);
		}

		template<typename desc, typename Functor, typename... Args>
		template<size_t... I>
		auto Function<desc, Functor, Args...>::scalarImpl(std::index_sequence<I...>,
														  size_t index) const -> Scalar {
			return m_functor((std::get<I>(m_args).scalar(index))...);
		}
	} // namespace detail

	template<typename... Inputs>
	using FunctionRef = detail::Function<Inputs...>;
} // namespace librapid

#endif // LIBRAPID_ARRAY_FUNCTION_HPP