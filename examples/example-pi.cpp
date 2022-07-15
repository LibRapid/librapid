#include <librapid/librapid.hpp>

namespace lrc = librapid;
using namespace librapid::suffix;
using half = lrc::extended::float16_t;

int main() {
	int64_t digits = 1000000;

	lrc::prec(digits);
	fmt::print("Calculating PI\n");
	auto chud = lrc::Chudnovsky(digits);
	auto pi	  = chud.pi();

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

	fmt::print("Inserting new lines\n");
	int64_t lineWidth = 100;
	int64_t index	  = lineWidth - 1;
	int64_t inserted  = 0;
	while (index < piStr.length()) {
		piStr.insert(index, "\n  ");
		index += lineWidth;
		++inserted;

		/*
		if (inserted % 500 == 0) {
			fmt::print("[ PROGRESS ] {:>5.2f}%\n", ((double)index / (double)piStr.length()) * 100);
		}
		*/
	}

	fmt::print("Writing to file...\n");
	std::fstream file;
	file.open("pi-librapid.txt", std::ios::out);
	fmt::print(file, "Pi to {} digits:\n{}", digits, piStr);
	file.close();

	return 0;
}
