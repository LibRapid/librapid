#include <librapid>

namespace lrc = librapid;

auto main() -> int {
	fmt::print("LibRapid Example -- Complex 1\n");

	fmt::print("sqrt(-1) = {}\n", lrc::sqrt(lrc::Complex(-1)));

	lrc::Complex<double> z(1, 2);
	fmt::print("z = {}\n", z);

	fmt::print("z + z = {}\n", z + z);
	fmt::print("z * z = {}\n", z * z);
	fmt::print("sin(z) = {}\n", lrc::sin(z));
	fmt::print("log(z) = {}\n", lrc::log(z));
	fmt::print("exp(z) = {}\n", lrc::exp(z));

	return 0;
}