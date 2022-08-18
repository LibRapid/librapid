#include <librapid>

namespace lrc = librapid;

int main() {
	// Test the most fundamental operations in Arrays. If these don't work, nothing else can
	// realistically be tested. If these work, however, we can test the other features in relation
	// to these operations

	// Initial Assumptions:
	// We must assume that an Array object can be correctly allocated on the corresponding device
	// and with the correct datatype.

	bool passed = true;

#define TEST(TYPE, DEVICE)                                                                         \
	{                                                                                              \
		lrc::Array<TYPE, lrc::device::DEVICE> test1d(lrc::Extent(5));                              \
		lrc::Array<TYPE, lrc::device::DEVICE> test2d(lrc::Extent(5, 5));                           \
                                                                                                   \
		for (int i = 0; i < 5; ++i) {                                                              \
			test1d(i) = i;                                                                         \
			for (int j = 0; j < 5; ++j) { test2d(i, j) = j + i * 5; }                              \
		}                                                                                          \
                                                                                                   \
		auto directAccess =                                                                        \
		  lrc::test::Test([&]() {                                                                  \
			  return std::vector<TYPE>({test2d(0, 0), test2d(0, 1), test2d(1, 0), test2d(4, 4)});  \
		  })                                                                                       \
			.name(fmt::format(                                                                     \
			  "Direct Access ({} -> {})", lrc::internal::traits<TYPE>::Name, STRINGIFY(DEVICE)))   \
			.description("Directly access a single element in an Array")                           \
			.expect(std::vector<TYPE>({0, 1, 5, 24}));                                             \
                                                                                                   \
		auto indirectAccess1 =                                                                     \
		  lrc::test::Test([&]() { return (TYPE)test2d[0][0]; })                                    \
			.name(fmt::format("Indirect Access Test 1 ({} -> {})",                                 \
							  lrc::internal::traits<TYPE>::Name,                                   \
							  STRINGIFY(DEVICE)))                                                  \
			.description("Accessing an Array element via repeated subscripting")                   \
			.expect(0);                                                                            \
                                                                                                   \
		auto indirectAccess2 =                                                                     \
		  lrc::test::Test([&]() { return (TYPE)test2d[1][4]; })                                    \
			.name(fmt::format("Indirect Access Test 2 ({} -> {})",                                 \
							  lrc::internal::traits<TYPE>::Name,                                   \
							  STRINGIFY(DEVICE)))                                                  \
			.description("Accessing an Array element via repeated subscripting")                   \
			.expect(9);                                                                            \
                                                                                                   \
		auto stringify1 =                                                                          \
		  lrc::test::Test(                                                                         \
			[&]() { return test1d.str(std::is_floating_point_v<TYPE> ? "{:.0f}" : "{}"); })        \
			.name(fmt::format(                                                                     \
			  "Stringify 1D ({} -> {})", lrc::internal::traits<TYPE>::Name, STRINGIFY(DEVICE)))    \
			.description("Get a string representation of an Array")                                \
			.expect(std::string("[0 1 2 3 4]"));                                                   \
                                                                                                   \
		auto stringify2 =                                                                          \
		  lrc::test::Test(                                                                         \
			[&]() { return test2d.str(std::is_floating_point_v<TYPE> ? "{:.0f}" : "{}"); })        \
			.name(fmt::format(                                                                     \
			  "Stringify 2D ({} -> {})", lrc::internal::traits<TYPE>::Name, STRINGIFY(DEVICE)))    \
			.description("Get a string representation of an Array")                                \
			.expect(std::string("[[ 0  1  2  3  4]\n"                                              \
								" [ 5  6  7  8  9]\n"                                              \
								" [10 11 12 13 14]\n"                                              \
								" [15 16 17 18 19]\n"                                              \
								" [20 21 22 23 24]]"));                                            \
                                                                                                   \
		directAccess.run();                                                                        \
		indirectAccess1.run();                                                                     \
		indirectAccess2.run();                                                                     \
		stringify1.run();                                                                          \
		stringify2.run();                                                                          \
                                                                                                   \
		if (!directAccess.passed() || !indirectAccess1.passed() || !indirectAccess2.passed() ||    \
			!stringify1.passed() || !stringify2.passed())                                          \
			passed = false;                                                                        \
	}

	TEST(int8_t, CPU)
	TEST(int16_t, CPU)
	TEST(int32_t, CPU)
	TEST(int64_t, CPU)
	TEST(float, CPU)
	TEST(double, CPU)

#if defined(LIBRAPID_HAS_CUDA)
	TEST(int8_t, GPU)
	TEST(int16_t, GPU)
	TEST(int32_t, GPU)
	TEST(int64_t, GPU)
	TEST(float, GPU)
	TEST(double, GPU)
#endif

	if (passed) return 0;
	return 1;
}