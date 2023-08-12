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
                using T1 = typename TypeInfo<std::decay_t<First>>::Scalar;
                return TypeInfo<std::decay_t<First>>::allowVectorisation &&
                       checkAllowVectorisation<T...>() &&
                       (std::is_same_v<T1, typename TypeInfo<std::decay_t<T>>::Scalar> && ...);
            }
        }

        template<typename First, typename... Rest>
        constexpr auto commonBackend() {
            using FirstBackend = typename TypeInfo<std::decay_t<First>>::Backend;
            if constexpr (sizeof...(Rest) == 0) {
                return FirstBackend {};
            } else {
                using RestBackend = decltype(commonBackend<Rest...>());
                if constexpr (std::is_same_v<FirstBackend, backend::OpenCLIfAvailable> ||
                              std::is_same_v<RestBackend, backend::OpenCLIfAvailable>) {
                    return backend::OpenCLIfAvailable {};
                } else if constexpr (std::is_same_v<FirstBackend, backend::CUDAIfAvailable> ||
                                     std::is_same_v<RestBackend, backend::CUDAIfAvailable>) {
                    return backend::CUDAIfAvailable {};
                } else {
                    return backend::CPU {};
                }
            }
        }

        template<typename desc, typename Functor_, typename... Args>
        struct TypeInfo<::librapid::detail::Function<desc, Functor_, Args...>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::ArrayFunction;
            using Scalar                               = decltype(std::declval<Functor_>()(
              std::declval<typename TypeInfo<std::decay_t<Args>>::Scalar>()...));
            using Backend                              = decltype(commonBackend<Args...>());

            static constexpr bool allowVectorisation = checkAllowVectorisation<Args...>();

            static constexpr bool supportsArithmetic = TypeInfo<Scalar>::supportsArithmetic;
            static constexpr bool supportsLogical    = TypeInfo<Scalar>::supportsLogical;
            static constexpr bool supportsBinary     = TypeInfo<Scalar>::supportsBinary;
        };

        LIBRAPID_DEFINE_AS_TYPE(typename desc COMMA typename Functor_ COMMA typename... Args,
                                ::librapid::detail::Function<desc COMMA Functor_ COMMA Args...>);
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

        template<typename desc, typename Functor_, typename... Args>
        class Function {
        public:
            using Type       = Function<desc, Functor_, Args...>;
            using Functor    = Functor_;
            using ShapeType  = Shape<size_t, 32>;
            using StrideType = ShapeType;
            using Scalar     = typename typetraits::TypeInfo<Type>::Scalar;
            using Backend    = typename typetraits::TypeInfo<Type>::Backend;
            using Packet     = typename typetraits::TypeInfo<Scalar>::Packet;
            using Iterator   = detail::ArrayIterator<Function>;

            using Descriptor = desc;
            static constexpr bool argsAreSameType =
              !std::is_same_v<decltype(scalarTypesAreSame<Args...>()), std::false_type>;

            Function() = default;

            /// Constructs a Function from a functor and arguments.
            /// \param functor The functor to use.
            /// \param args The arguments to use.
            LIBRAPID_ALWAYS_INLINE explicit Function(const Functor &functor, const Args &...args);

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

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index) const;

            /// Evaluates the function at the given index, returning a Packet result.
            /// \param index The index to evaluate at.
            /// \return The result of the function (vectorized).
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packet(size_t index) const;

            /// Evaluates the function at the given index, returning a Scalar result.
            /// \param index The index to evaluate at.
            /// \return The result of the function (scalar).
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar scalar(size_t index) const;

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator begin() const;
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator end() const;

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
        Function<desc, Functor, Args...>::Function(const Functor &functor, const Args &...args) :
                m_functor(functor), m_args(args...) {}

        template<typename desc, typename Functor, typename... Args>
        auto Function<desc, Functor, Args...>::shape() const {
            return typetraits::TypeInfo<Functor>::getShape(m_args);
        }

        template<typename desc, typename Functor, typename... Args>
        auto &Function<desc, Functor, Args...>::args() const {
            return m_args;
        }

        template<typename desc, typename Functor, typename... Args>
        auto Function<desc, Functor, Args...>::operator[](int64_t index) const {
            return array::ArrayView(*this)[index];
        }

        template<typename desc, typename Functor, typename... Args>
        auto Function<desc, Functor, Args...>::eval() const {
            auto res = Array<Scalar, Backend>(shape());
            res      = *this;
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
            return m_functor(scalarExtractor(std::get<I>(m_args), index)...);
        }

        template<typename desc, typename Functor, typename... Args>
        auto Function<desc, Functor, Args...>::begin() const -> Iterator {
            return Iterator(*this, 0);
        }

        template<typename desc, typename Functor, typename... Args>
        auto Function<desc, Functor, Args...>::end() const -> Iterator {
            return Iterator(*this, shape()[0]);
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

LIBRAPID_SIMPLE_IO_NORANGE(typename desc COMMA typename Functor COMMA typename... Args,
                           librapid::detail::Function<desc COMMA Functor COMMA Args...>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_FUNCTION_HPP