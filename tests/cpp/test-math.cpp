#include <librapid>

namespace lrc = librapid;

template<typename T>
bool isClose(T a, T b) {
	return std::abs(a - b) <= lrc::internal::traits<T>::epsilon();
}

int main() {
	// Test basic arithmetic functionality -- addition, subtraction, multiplication, division, etc.
	//
	// NOTE:
	// Don't test for anything below int32, as they get promoted to int32 and this causes compile
	// errors

	bool passed = true;

#define TYPE float

	// Test product
	auto roundingTest = lrc::test::Test([&]() {
							return std::vector<TYPE>({// Normal
													  lrc::round(TYPE(0)),
													  lrc::round(TYPE(0.25)),
													  lrc::round(TYPE(0.5)),
													  lrc::round(TYPE(0.75)),
													  lrc::round(TYPE(0.499999)),
													  lrc::round(TYPE(5)),
													  lrc::round(TYPE(5.5)),
													  // 3 dp
													  lrc::round(TYPE(0), 3),
													  lrc::round(TYPE(0.25), 3),
													  lrc::round(TYPE(0.5), 3),
													  lrc::round(TYPE(0.75), 3),
													  lrc::round(TYPE(0.499999), 3),
													  lrc::round(TYPE(5), 3),
													  lrc::round(TYPE(5.5), 3),
													  // Negative
													  lrc::round(TYPE(-0), 3),
													  lrc::round(TYPE(-0.25), 3),
													  lrc::round(TYPE(-0.5), 3),
													  lrc::round(TYPE(-0.75), 3),
													  lrc::round(TYPE(-0.499999), 3),
													  lrc::round(TYPE(-5), 3),
													  lrc::round(TYPE(-5.5), 3)});
						})
						  .name(fmt::format("Rounding Test [ {} ]", STRINGIFY(TYPE)))
						  .description("Round a number to a given number of decimal places")
						  .expect(std::vector<TYPE>({// Normal
													 TYPE(0),
													 TYPE(0),
													 TYPE(1),
													 TYPE(1),
													 TYPE(0),
													 TYPE(5),
													 TYPE(6),
													 // 3 dp
													 TYPE(0.000),
													 TYPE(0.250),
													 TYPE(0.500),
													 TYPE(0.750),
													 TYPE(0.500),
													 TYPE(5.000),
													 TYPE(5.500),
													 // Negative
													 TYPE(-0.000),
													 TYPE(-0.250),
													 TYPE(-0.500),
													 TYPE(-0.750),
													 TYPE(-0.500),
													 TYPE(-5.000),
													 TYPE(-5.500)}));

	auto roundToTest = lrc::test::Test([&]() {
						   return std::vector<TYPE>({
							 // Integers
							 lrc::roundTo(TYPE(0), TYPE(0)),
							 lrc::roundTo(TYPE(1), TYPE(5)),
							 lrc::roundTo(TYPE(3), TYPE(6)),
							 lrc::roundTo(TYPE(10), TYPE(3)),
							 // Floating
							 lrc::roundTo(TYPE(0.5), TYPE(0.75)),
							 lrc::roundTo(TYPE(1.25), TYPE(2)),
							 lrc::roundTo(TYPE(5.75), TYPE(3)),
							 lrc::roundTo(TYPE(0.125), TYPE(0.0625)),
							 // Negative
							 lrc::roundTo(TYPE(-0.5), TYPE(0.75)),
							 lrc::roundTo(TYPE(-1.25), TYPE(2)),
							 lrc::roundTo(TYPE(-5.75), TYPE(3)),
							 lrc::roundTo(TYPE(-0.125), TYPE(0.0625)),
						   });
					   })
						 .name(fmt::format("Rounding To Test [ {} ]", STRINGIFY(TYPE)))
						 .description("Round a number to the nearest multiple of a number")
						 .expect(std::vector<TYPE>({// Integers
													TYPE(0),
													TYPE(0),
													TYPE(6),
													TYPE(9),
													// Floating
													TYPE(0.75),
													TYPE(2),
													TYPE(6),
													TYPE(0.125),
													// Negative
													TYPE(-0.75),
													TYPE(-2),
													TYPE(-6),
													TYPE(-0.125)}));

	roundingTest.run();
	roundToTest.run();

	if (!roundingTest.passed() || !roundToTest.passed()) { passed = false; }

	if (passed) return 0;
	return 1;
}