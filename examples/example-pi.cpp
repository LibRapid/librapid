#include <librapid>

namespace lrc = librapid;
using namespace librapid::suffix;

int main() {
	int64_t digits = 1000000;

	lrc::prec(digits);

	fmt::print("Calculating PI\n");
	auto pi	  = lrc::constPi();

	fmt::print("Stringifying Pi\n");
	std::string piStr = fmt::format("{:.1000000}", pi);

	fmt::print("Verifying...\n");
	std::string verifyCheck = // Taken from https://www.piday.org/million/
	  "3321272849194418437150696552087542450598956787961303311646283996346460422090106105779458151";
	auto verifyLen = verifyCheck.length();
	if (piStr.substr(piStr.length() - verifyLen, verifyLen) == verifyCheck) {
		fmt::print("Correct :)\n");
	} else {
		fmt::print(
		  "Pi was calculated incorrectly... Please raise this as an issue with all configuration "
		  "settings!\n");
		fmt::print("Expected: {}\n", verifyCheck);
		fmt::print("Result  : {}\n", piStr.substr(piStr.length() - verifyLen, verifyLen));
		return 0;
	}

	int64_t digitsPerBlock = 10;
	int64_t blocksPerLine  = 5;
	int64_t digitsPerLine  = digitsPerBlock * blocksPerLine;

	fmt::print("Writing to file...\n");
	std::fstream file;
	file.open("pi-librapid.txt", std::ios::out);
	fmt::print(file, "Pi to {} digits:\n3.\n", digits, piStr);

	for (auto it = piStr.begin() + 2; it < piStr.end(); it += digitsPerLine) {
		fmt::print(file, " ");
		for (int64_t i = 0; i < blocksPerLine; ++i) {
			fmt::print(
			  file, " {}", std::string(it + digitsPerBlock * i, it + digitsPerBlock * (i + 1)));
		}
		fmt::print(file, "\n");
	}

	file.close();

	return 0;
}
