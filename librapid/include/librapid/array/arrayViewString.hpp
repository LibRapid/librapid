#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP

namespace librapid {
	namespace detail {
		template<typename T>
		LIBRAPID_INLINE std::pair<size_t, size_t> countWidth(const T &val,
															 const std::string &format) {
			std::string str = fmt::format(format, val);
			return {str.size(), 0};
		}

		template<>
		LIBRAPID_INLINE std::pair<size_t, size_t> countWidth(const float &val,
															 const std::string &format) {
			std::string str = fmt::format(format, val);
			auto point		= str.find('.');
			if (point == std::string::npos) { return {str.size(), 0}; }
			return {point, str.size() - point - 1};
		}

		template<typename T>
		std::vector<std::pair<int64_t, int64_t>> countColumnWidths(const array::ArrayView<T> &view,
																   const std::string &format) {
			if (view.ndim() == 0) {
				// Scalar
				return {{countWidth(view.scalar(0), format)}};
			} else if (view.ndim() == 1) {
				// Vector
				std::vector<std::pair<int64_t, int64_t>> widths(view.shape()[0]);
				for (int64_t i = 0; i < view.shape()[0]; ++i) {
					widths[i] = countWidth(view.scalar(i), format);
				}
				return widths;
			} else {
				// General
				std::vector<std::pair<int64_t, int64_t>> widths =
				  countColumnWidths(view[0], format);
				for (int64_t i = 1; i < view.shape()[0]; ++i) {
					auto subWidths = countColumnWidths(view[i], format);
					for (int64_t j = 0; j < widths.size(); ++j) {
						widths[j].first	 = ::librapid::max(widths[j].first, subWidths[j].first);
						widths[j].second = ::librapid::max(widths[j].second, subWidths[j].second);
					}
				}
				return widths;
			}
		}

		template<typename T>
		std::string arrayViewToString(const array::ArrayView<T> &view, const std::string &format,
									  const std::vector<std::pair<int64_t, int64_t>> &widths,
									  int64_t indent) {
			if (view.ndim() == 0) { return fmt::format(format, view.scalar(0)); }

			if (view.ndim() == 1) {
				std::string str = "[";
				for (int64_t i = 0; i < view.shape()[0]; i++) {
					std::pair<int64_t, int64_t> width = detail::countWidth(view.scalar(i), format);
					str += fmt::format("{:>{}}{}{:>{}}",
									   "",
									   widths[i].first - width.first,
									   view.scalar(i),
									   "",
									   widths[i].second - width.second + (width.second == 0));
					if (i != view.shape()[0] - 1) { str += ", "; }
				}
				str += "]";
				return str;
			}

			std::string str = "[";
			for (int64_t i = 0; i < view.shape()[0]; i++) {
				if (i > 0) str += std::string(indent + 1, ' ');
				str += arrayViewToString(view[i], format, widths, indent + 1);
				if (i != view.shape()[0] - 1) { str += "\n"; }
			}
			str += "]";
			return str;
		}
	} // namespace detail

	namespace array {
		template<typename T>
		auto ArrayView<T>::str(const std::string &format) const -> std::string {
			std::vector<std::pair<int64_t, int64_t>> widths =
			  detail::countColumnWidths(*this, format);
			return detail::arrayViewToString(*this, format, widths, 0);
		}
	} // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP