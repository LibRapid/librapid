#ifndef LIBRAPID_MATH_COMPILE_TIME_HPP
#define LIBRAPID_MATH_COMPILE_TIME_HPP

namespace librapid {
	template<size_t First, size_t... Rest>
	constexpr size_t product() {
		if constexpr (sizeof...(Rest) == 0) {
			return First;
		} else {
			return First * product<Rest...>();
		}
	}
} // namespace librapid

#endif // LIBRAPID_MATH_COMPILE_TIME_HPP