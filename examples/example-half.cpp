#include <librapid>

namespace lrc = librapid;

int main() {
	fmt::print("sin(0.5h) = {}\n", lrc::sin(lrc::half(0.5)));
	fmt::print("cos(0.5h) = {}\n", lrc::cos(lrc::half(0.5)));
	fmt::print("tan(0.5h) = {}\n", lrc::tan(lrc::half(0.5)));
	fmt::print("asin(0.5h) = {}\n", lrc::asin(lrc::half(0.5)));
	fmt::print("acos(0.5h) = {}\n", lrc::acos(lrc::half(0.5)));
	fmt::print("atan(0.5h) = {}\n", lrc::atan(lrc::half(0.5)));
	fmt::print("sinh(0.5h) = {}\n", lrc::sinh(lrc::half(0.5)));
	fmt::print("cosh(0.5h) = {}\n", lrc::cosh(lrc::half(0.5)));
	fmt::print("tanh(0.5h) = {}\n", lrc::tanh(lrc::half(0.5)));
	fmt::print("asinh(0.5h) = {}\n", lrc::asinh(lrc::half(0.5)));
	fmt::print("acosh(0.5h) = {}\n", lrc::acosh(lrc::half(0.5)));
	fmt::print("atanh(0.5h) = {}\n", lrc::atanh(lrc::half(0.5)));
	fmt::print("exp(0.5h) = {}\n", lrc::exp(lrc::half(0.5)));
	fmt::print("log(0.5h) = {}\n", lrc::log(lrc::half(0.5)));
	fmt::print("log10(0.5h) = {}\n", lrc::log10(lrc::half(0.5)));
	fmt::print("sqrt(0.5h) = {}\n", lrc::sqrt(lrc::half(0.5)));
	fmt::print("ceil(0.5h) = {}\n", lrc::ceil(lrc::half(0.5)));
	fmt::print("floor(0.5h) = {}\n", lrc::floor(lrc::half(0.5)));
	fmt::print("round(0.1h, 3dp) = {}\n", lrc::round(lrc::half(0.1), 3));
	fmt::print("fmod(0.5h, 0.25h) = {}\n", lrc::fmod(lrc::half(0.5), lrc::half(0.25)));
	fmt::print("pow(0.5h, 0.25h) = {}\n", lrc::pow(lrc::half(0.5), lrc::half(0.25)));
	fmt::print("atan2(0.5h, 0.25h) = {}\n", lrc::atan2(lrc::half(0.5), lrc::half(0.25)));

	return 0;
}