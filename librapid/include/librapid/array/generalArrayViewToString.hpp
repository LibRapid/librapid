#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP

namespace librapid {
	namespace detail {
		template<size_t N, typename Val>
		std::pair<int64_t, int64_t> alignment(const char (&formatString)[N], const Val &value) {
			std::string tmpFormat = fmt::format("{{:{}}}", formatString);
			std::string formatted = fmt::vformat(tmpFormat, fmt::make_format_args(value));

			if constexpr (std::is_integral_v<std::decay_t<Val>>) {
				return std::make_pair(formatted.length(), 0);
			} else if constexpr (std::is_floating_point_v<std::decay_t<Val>>) {
				auto point = formatted.find('.');
				if (point == std::string::npos) {
					return std::make_pair(formatted.length(), 0);
				} else {
					return std::make_pair(point, formatted.length() - point);
				}
			}

			return std::make_pair(0, 0);
		}

		template<typename ArrayViewType, typename ArrayViewShapeType, size_t N>
		void generalArrayViewToStringColWidthFinder(
		  const array::GeneralArrayView<ArrayViewType, ArrayViewShapeType> &view,
		  const char (&formatString)[N], std::vector<std::pair<int64_t, int64_t>> &alignments) {
			if (view.ndim() == 1) {
				for (int64_t i = 0; i < static_cast<int64_t>(view.shape()[0]); i++) {
					auto alignmentPair = alignment(formatString, view.scalar(i));
					if (i >= static_cast<int64_t>(alignments.size())) {
						alignments.push_back(alignmentPair);
					} else {
						alignments[i].first =
						  ::librapid::max(alignments[i].first, alignmentPair.first);
						alignments[i].second =
						  ::librapid::max(alignments[i].second, alignmentPair.second);
					}
				}
			} else if (view.ndim() > 1) {
				for (int64_t i = 0; i < static_cast<int64_t>(view.shape()[0]); i++) {
					generalArrayViewToStringColWidthFinder(view[i], formatString, alignments);
				}
			}
		}

		template<typename ArrayViewType, typename ArrayViewShapeType, typename T, typename Char,
				 size_t N, typename Ctx>
		void generalArrayViewToStringImpl(
		  const array::GeneralArrayView<ArrayViewType, ArrayViewShapeType> &view,
		  const fmt::formatter<T, Char> &formatter, char bracket, char separator,
		  const char (&formatString)[N], int64_t indent, Ctx &ctx,
		  const std::vector<std::pair<int64_t, int64_t>> &alignments) {
			char bracketCharOpen, bracketCharClose;

			switch (bracket) {
				case 'r':
					bracketCharOpen	 = '(';
					bracketCharClose = ')';
					break;
				case 's':
					bracketCharOpen	 = '[';
					bracketCharClose = ']';
					break;
				case 'c':
					bracketCharOpen	 = '{';
					bracketCharClose = '}';
					break;
				case 'a':
					bracketCharOpen	 = '<';
					bracketCharClose = '>';
					break;
				case 'p':
					bracketCharOpen	 = '|';
					bracketCharClose = '|';
					break;
				default:
					bracketCharOpen	 = '[';
					bracketCharClose = ']';
					break;
			}

			// Separator char is already the correct character

			if (view.ndim() == 0) {
				formatter.format(view.scalar(0), ctx);
			} else if (view.ndim() == 1) {
				fmt::format_to(ctx.out(), "{}", bracketCharOpen);
				for (int64_t i = 0; i < static_cast<int64_t>(view.shape()[0]); i++) {
					auto columnAlignment = alignments[i];
					auto valueSize		 = alignment(formatString, view.scalar(i));
					int64_t pre	 = max(columnAlignment.first - valueSize.first, 0),
							post = max(columnAlignment.second - valueSize.second, 0);

					fmt::format_to(ctx.out(), "{}", std::string(pre, ' '));
					formatter.format(view.scalar(i), ctx);
					fmt::format_to(ctx.out(), "{}", std::string(post, ' '));

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
					generalArrayViewToStringImpl(view[i],
												 formatter,
												 bracket,
												 separator,
												 formatString,
												 indent + 1,
												 ctx,
												 alignments);
					if (i != view.shape()[0] - 1) {
						if (separator == ' ') {
							fmt::format_to(ctx.out(), "\n");
						} else {
							fmt::format_to(ctx.out(), "{}\n", separator);
						}
						if (view.ndim() > 2) { fmt::format_to(ctx.out(), "\n"); }
					}
				}
				fmt::format_to(ctx.out(), "{}", bracketCharClose);
			}
		}

		template<typename ArrayViewType, typename ArrayViewShapeType, typename T, typename Char,
				 size_t N, typename Ctx>
		void generalArrayViewToString(
		  const array::GeneralArrayView<ArrayViewType, ArrayViewShapeType> &view,
		  const fmt::formatter<T, Char> &formatter, char bracket, char separator,
		  const char (&formatString)[N], int64_t indent, Ctx &ctx) {
			std::vector<std::pair<int64_t, int64_t>> alignments;
			generalArrayViewToStringColWidthFinder(view, formatString, alignments);
			generalArrayViewToStringImpl(
			  view, formatter, bracket, separator, formatString, indent, ctx, alignments);
		}
	} // namespace detail

	namespace array {
		template<typename ArrayViewType, typename ArrayViewShapeType>
		template<typename T, typename Char, size_t N, typename Ctx>
		void GeneralArrayView<ArrayViewType, ArrayViewShapeType>::str(
		  const fmt::formatter<T, Char> &format, char bracket, char separator,
		  const char (&formatString)[N], Ctx &ctx) const {
			detail::generalArrayViewToString(
			  *this, format, bracket, separator, formatString, 0, ctx);
		}
	} // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP