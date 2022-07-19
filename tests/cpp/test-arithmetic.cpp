#include <librapid>

namespace lrc = librapid;

int main() {
	// Test basic arithmetic functionality -- addition, subtraction, multiplication, division, etc.
	//
	// NOTE:
	// Don't test for anything below int32, as they get promoted to int32 and this causes compile
	// errors

	bool passed = true;

#define TEST(FUNC, TYPE, DEVICE)                                                                   \
	{                                                                                              \
		/* This ensures that we cannot use vectorised instructions on the entire array, */         \
		/* improving code coverage and forcing small, final loops to be run */                     \
		int64_t packetWidth = lrc::internal::traits<TYPE>::PacketWidth;                            \
		int64_t size		= lrc::roundUpTo(64, packetWidth) + 1;                                 \
                                                                                                   \
		lrc::Array<TYPE, lrc::device::DEVICE> test1(lrc::Extent(size, size));                      \
		lrc::Array<TYPE, lrc::device::DEVICE> test2(lrc::Extent(size, size));                      \
                                                                                                   \
		for (int64_t i = 0; i < size; ++i) {                                                       \
			for (int64_t j = 0; j < size; ++j) {                                                   \
				test1(i, j) = j + i * size;                                                        \
				test2(j, i) = j + i * size;                                                        \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto lazy = lrc::test::Test([&]() {                                                        \
						auto res = OP_##FUNC(test1, test2);                                        \
						return std::vector<TYPE>({res(0, 0),                                       \
												  res(0, 1),                                       \
												  res(1, 0),                                       \
												  res(10, 10),                                     \
												  res(size - 1, size - 2),                         \
												  res(size - 2, size - 1),                         \
												  res(size - 1, size - 1)});                       \
					})                                                                             \
					  .name(fmt::format("Lazy Evaluation  [{}]  ({} -> {})",                       \
										STRINGIFY(OP_##FUNC),                                      \
										STRINGIFY(TYPE),                                           \
										STRINGIFY(DEVICE)))                                        \
					  .description("Add two arrays together, testing the lazy-evaluated result")   \
					  .expect(std::vector<TYPE>({                                                  \
						OP_##FUNC(test1(0, 0), test2(0, 0)),                                       \
						OP_##FUNC(test1(0, 1), test2(0, 1)),                                       \
						OP_##FUNC(test1(1, 0), test2(1, 0)),                                       \
						OP_##FUNC(test1(10, 10), test2(10, 10)),                                   \
						OP_##FUNC(test1(size - 1, size - 2), test2(size - 1, size - 2)),           \
						OP_##FUNC(test1(size - 2, size - 1), test2(size - 2, size - 1)),           \
						OP_##FUNC(test1(size - 1, size - 1), test2(size - 1, size - 1)),           \
					  }));                                                                         \
                                                                                                   \
		auto force = lrc::test::Test([&]() {                                                       \
						 lrc::Array<TYPE, lrc::device::DEVICE> res = OP_##FUNC(test1, test2);      \
						 return std::vector<TYPE>({res(0, 0),                                      \
												   res(0, 1),                                      \
												   res(1, 0),                                      \
												   res(10, 10),                                    \
												   res(size - 1, size - 2),                        \
												   res(size - 2, size - 1),                        \
												   res(size - 1, size - 1)});                      \
					 })                                                                            \
					   .name(fmt::format("Force Evaluation  [{}]  ({} -> {})",                     \
										 STRINGIFY(OP_##FUNC),                                     \
										 STRINGIFY(TYPE),                                          \
										 STRINGIFY(DEVICE)))                                       \
					   .description("Add two arrays together, testing the evaluated result")       \
					   .expect(std::vector<TYPE>({                                                 \
						 OP_##FUNC(test1(0, 0), test2(0, 0)),                                      \
						 OP_##FUNC(test1(0, 1), test2(0, 1)),                                      \
						 OP_##FUNC(test1(1, 0), test2(1, 0)),                                      \
						 OP_##FUNC(test1(10, 10), test2(10, 10)),                                  \
						 OP_##FUNC(test1(size - 1, size - 2), test2(size - 1, size - 2)),          \
						 OP_##FUNC(test1(size - 2, size - 1), test2(size - 2, size - 1)),          \
						 OP_##FUNC(test1(size - 1, size - 1), test2(size - 1, size - 1)),          \
					   }));                                                                        \
                                                                                                   \
		auto lazyScalar =                                                                          \
		  lrc::test::Test([&]() {                                                                  \
			  auto res = OP_##FUNC(test1, 100);                                                    \
			  return std::vector<TYPE>({res(0, 0),                                                 \
										res(0, 1),                                                 \
										res(1, 0),                                                 \
										res(10, 10),                                               \
										res(size - 1, size - 2),                                   \
										res(size - 2, size - 1),                                   \
										res(size - 1, size - 1)});                                 \
		  })                                                                                       \
			.name(fmt::format("Array Scalar  [{}]  ({} -> {})",                                    \
							  STRINGIFY(OP_##FUNC),                                                \
							  STRINGIFY(TYPE),                                                     \
							  STRINGIFY(DEVICE)))                                                  \
			.description("Add a scalar to an Array, testing the lazy-evaluated result")            \
			.expect(std::vector<TYPE>({                                                            \
			  OP_##FUNC(test1(0, 0), 100),                                                         \
			  OP_##FUNC(test1(0, 1), 100),                                                         \
			  OP_##FUNC(test1(1, 0), 100),                                                         \
			  OP_##FUNC(test1(10, 10), 100),                                                       \
			  OP_##FUNC(test1(size - 1, size - 2), 100),                                           \
			  OP_##FUNC(test1(size - 2, size - 1), 100),                                           \
			  OP_##FUNC(test1(size - 1, size - 1), 100),                                           \
			}));                                                                                   \
                                                                                                   \
		auto lazyScalar2 = lrc::test::Test([&]() {                                                 \
							   auto res = OP_##FUNC(100, test1);                                   \
							   return std::vector<TYPE>({res(0, 0),                                \
														 res(0, 1),                                \
														 res(1, 0),                                \
														 res(10, 10),                              \
														 res(size - 1, size - 2),                  \
														 res(size - 2, size - 1),                  \
														 res(size - 1, size - 1)});                \
						   })                                                                      \
							 .name(fmt::format("Scalar Array [{}]  ({} -> {})",                    \
											   STRINGIFY(OP_##FUNC),                               \
											   STRINGIFY(TYPE),                                    \
											   STRINGIFY(DEVICE)))                                 \
							 .description("Add an Array to a scalar, testing the lazy result")     \
							 .expect(std::vector<TYPE>({                                           \
							   OP_##FUNC(100, test1(0, 0)),                                        \
							   OP_##FUNC(100, test1(0, 1)),                                        \
							   OP_##FUNC(100, test1(1, 0)),                                        \
							   OP_##FUNC(100, test1(10, 10)),                                      \
							   OP_##FUNC(100, test1(size - 1, size - 2)),                          \
							   OP_##FUNC(100, test1(size - 2, size - 1)),                          \
							   OP_##FUNC(100, test1(size - 1, size - 1)),                          \
							 }));                                                                  \
                                                                                                   \
		lazy.run();                                                                                \
		force.run();                                                                               \
		lazyScalar.run();                                                                          \
		lazyScalar2.run();                                                                         \
                                                                                                   \
		if (!lazy.passed() || !force.passed() || !lazyScalar.passed() || !lazyScalar2.passed())    \
			passed = false;                                                                        \
	}

	auto OP_ADD = [&](auto x, auto y) { return x + y; };
	auto OP_SUB = [&](auto x, auto y) { return x + y; };
	auto OP_MUL = [&](auto x, auto y) { return x + y; };
	auto OP_DIV = [&](auto x, auto y) { return x + y; };

	TEST(ADD, int32_t, CPU)
	TEST(ADD, int64_t, CPU)
	TEST(ADD, float, CPU)
	TEST(ADD, double, CPU)

	TEST(SUB, int32_t, CPU)
	TEST(SUB, int64_t, CPU)
	TEST(SUB, float, CPU)
	TEST(SUB, double, CPU)

	TEST(MUL, int32_t, CPU)
	TEST(MUL, int64_t, CPU)
	TEST(MUL, float, CPU)
	TEST(MUL, double, CPU)

	TEST(DIV, int32_t, CPU)
	TEST(DIV, int64_t, CPU)
	TEST(DIV, float, CPU)
	TEST(DIV, double, CPU)

#if defined(LIBRAPID_HAS_CUDA)
	TEST(ADD, int32_t, GPU)
	TEST(ADD, int64_t, GPU)
	TEST(ADD, float, GPU)
	TEST(ADD, double, GPU)

	TEST(SUB, int32_t, GPU)
	TEST(SUB, int64_t, GPU)
	TEST(SUB, float, GPU)
	TEST(SUB, double, GPU)

	TEST(MUL, int32_t, GPU)
	TEST(MUL, int64_t, GPU)
	TEST(MUL, float, GPU)
	TEST(MUL, double, GPU)

	TEST(DIV, int32_t, GPU)
	TEST(DIV, int64_t, GPU)
	TEST(DIV, float, GPU)
	TEST(DIV, double, GPU)
#endif

	if (passed) return 0;
	return 1;
}