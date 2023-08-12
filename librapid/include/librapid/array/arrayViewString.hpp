#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP

namespace librapid {
    namespace detail {
        template<typename ArrayViewType, typename T, typename Char, typename Ctx>
        void arrayViewToString(const array::ArrayView<ArrayViewType> &view,
                               const fmt::formatter<T, Char> &formatter, char bracket,
                               char separator, int64_t indent, Ctx &ctx) {
            char bracketCharOpen, bracketCharClose;

            switch (bracket) {
                case 'r':
                    bracketCharOpen  = '(';
                    bracketCharClose = ')';
                    break;
                case 's':
                    bracketCharOpen  = '[';
                    bracketCharClose = ']';
                    break;
                case 'c':
                    bracketCharOpen  = '{';
                    bracketCharClose = '}';
                    break;
                case 'a':
                    bracketCharOpen  = '<';
                    bracketCharClose = '>';
                    break;
                case 'p':
                    bracketCharOpen  = '|';
                    bracketCharClose = '|';
                    break;
                default:
                    bracketCharOpen  = '[';
                    bracketCharClose = ']';
                    break;
            }

            // Separator char is already the correct character

            if (view.ndim() == 0) {
                formatter.format(view.scalar(0), ctx);
            } else if (view.ndim() == 1) {
                fmt::format_to(ctx.out(), "{}", bracketCharOpen);
                for (int64_t i = 0; i < static_cast<int64_t>(view.shape()[0]); i++) {
                    formatter.format(view.scalar(i), ctx);
                    if (i != view.shape()[0] - 1) {
                        if (separator == ' ') {
                            fmt::format_to(ctx.out(), " ");
                        } else {
                            fmt::format_to(ctx.out(), "{} ", separator);
                        }
                    }
                }
                fmt::format_to(ctx.out(), "{}", bracketCharClose);
            } else {
                fmt::format_to(ctx.out(), "{}", bracketCharOpen);
                for (int64_t i = 0; i < static_cast<int64_t>(view.shape()[0]); i++) {
                    if (i > 0) fmt::format_to(ctx.out(), "{}", std::string(indent + 1, ' '));
                    arrayViewToString(view[i], formatter, bracket, separator, indent + 1, ctx);
                    if (i != view.shape()[0] - 1) {
                        fmt::format_to(ctx.out(), "{}\n", separator);
                        if (view.ndim() > 2) { fmt::format_to(ctx.out(), "\n"); }
                    }
                }
                fmt::format_to(ctx.out(), "{}", bracketCharClose);
            }
        }
    } // namespace detail

    namespace array {
        template<typename ArrayViewType>
        template<typename T, typename Char, typename Ctx>
        void ArrayView<ArrayViewType>::str(const fmt::formatter<T, Char> &format, char bracket,
                                           char separator, Ctx &ctx) const {
            detail::arrayViewToString(*this, format, bracket, separator, 0, ctx);
        }
    } // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP