#ifndef LIBRAPID_CORE_HELPER_MACROS
#define LIBRAPID_CORE_HELPER_MACROS

/*
 * Defines a set of basic macros for common uses
 */

#define COMMA ,

// Provide {fmt} printing capabilities
#define LIBRAPID_SIMPLE_IO_IMPL(TEMPLATE_, TYPE_)                                                  \
	template<TEMPLATE_>                                                                            \
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
				return fmt::format_to(ctx.out(), object.str());                                    \
			} catch (std::exception & e) { return fmt::format_to(ctx.out(), e.what()); }           \
		}                                                                                          \
	};                                                                                             \
                                                                                                   \
	template<TEMPLATE_>                                                                            \
	std::ostream &operator<<(std::ostream &os, const TYPE_ &object) {                              \
		os << object.str();                                                                        \
		return os;                                                                                 \
	}

#endif // LIBRAPID_CORE_HELPER_MACROS