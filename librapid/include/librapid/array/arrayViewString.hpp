#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP

namespace librapid::array {
	template<typename T>
	auto ArrayView<T>::str(const std::string &format) const -> std::string {
		if (ndim() == 0) { return fmt::format(format, scalar(0)); }
		if (ndim() == 1) {
			std::string str = "[";
			for (int64_t i = 0; i < m_shape[0]; i++) {
				str += fmt::format(format, scalar(i));
				if (i != m_shape[0] - 1) { str += ", "; }
			}
			str += "]";
			return str;
		}
	}
}

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_STRING_HPP