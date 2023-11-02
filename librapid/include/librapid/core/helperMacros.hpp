#ifndef LIBRAPID_CORE_HELPER_MACROS
#define LIBRAPID_CORE_HELPER_MACROS

/*
 * Defines a set of basic macros for common uses
 */

#define COMMA ,

#define LIBRAPID_SIMPLE_IO_IMPL(TEMPLATE_, TYPE_)                                                  \
	template<TEMPLATE_>                                                                            \
	struct fmt::formatter<TYPE_> {                                                                 \
		char formatStr[32] = {'{', ':'};                                                           \
                                                                                                   \
		constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {        \
			auto it		   = ctx.begin();                                                          \
			uint64_t index = 0;                                                                    \
			for (; it != ctx.end(); ++it) {                                                        \
				if (*it == '}') break;                                                             \
				formatStr[index++] += *it;                                                         \
			}                                                                                      \
			formatStr[index] = '}';                                                                \
			return it;                                                                             \
		}                                                                                          \
                                                                                                   \
		template<typename FormatContext>                                                           \
		auto format(const TYPE_ &object, FormatContext &ctx) {                                     \
			try {                                                                                  \
				return fmt::format_to(ctx.out(), object.str(formatStr));                           \
			} catch (std::exception & e) { return fmt::format_to(ctx.out(), e.what()); }           \
		}                                                                                          \
	};                                                                                             \
                                                                                                   \
	template<TEMPLATE_>                                                                            \
	std::ostream &operator<<(std::ostream &os, const TYPE_ &object) {                              \
		os << object.str();                                                                        \
		return os;                                                                                 \
	}

#define LIBRAPID_SIMPLE_IO_IMPL_NO_TEMPLATE(TYPE_)                                                 \
	template<>                                                                                     \
	struct fmt::formatter<TYPE_> {                                                                 \
		std::string formatStr = "{}";                                                              \
                                                                                                   \
		template<typename ParseContext>                                                            \
		constexpr auto parse(ParseContext &ctx) {                                                  \
			formatStr = "{:";                                                                      \
			auto it	  = ctx.begin();                                                               \
			for (; it != ctx.end(); ++it) {                                                        \
				if (*it == '}') break;                                                             \
				formatStr += *it;                                                                  \
			}                                                                                      \
			formatStr += "}";                                                                      \
			return it;                                                                             \
		}                                                                                          \
                                                                                                   \
		template<typename FormatContext>                                                           \
		auto format(const TYPE_ &object, FormatContext &ctx) {                                     \
			try {                                                                                  \
				return fmt::format_to(ctx.out(), object.str(formatStr));                           \
			} catch (std::exception & e) { return fmt::format_to(ctx.out(), e.what()); }           \
		}                                                                                          \
	};                                                                                             \
                                                                                                   \
	LIBRAPID_INLINE std::ostream &operator<<(std::ostream &os, const TYPE_ &object) {              \
		os << object.str();                                                                        \
		return os;                                                                                 \
	}

#if defined(FMT_RANGES_H_)
#	define LIBRAPID_SIMPLE_IO_NORANGE(TEMPLATE, TYPE)                                             \
		template<TEMPLATE, typename Char>                                                          \
		struct fmt::is_range<TYPE, Char> : std::false_type {};
#else
#	define LIBRAPID_SIMPLE_IO_NORANGE(TEMPLATE, TYPE)
#endif // FMT_RANGES_H_

namespace librapid::typetraits {
	template<typename T>
	struct IsLibRapidType : std::false_type {};
} // namespace librapid::typetraits

// Define a type as being part of librapid -- this should be contained in the typetraits namespace
#define LIBRAPID_DEFINE_AS_TYPE(TEMPLATE_, TYPE_)                                                  \
	template<TEMPLATE_>                                                                            \
	struct IsLibRapidType<TYPE_> : std::true_type {}

#define LIBRAPID_DEFINE_AS_TYPE_NO_TEMPLATE(TYPE_)                                                 \
	template<>                                                                                     \
	struct IsLibRapidType<TYPE_> : std::true_type {}

#endif // LIBRAPID_CORE_HELPER_MACROS