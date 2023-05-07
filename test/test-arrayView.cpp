#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;

// #define SCALAR float
// #define BACKEND lrc::backend::CPU

#define TEST_ARRAY_VIEW(SCALAR, BACKEND)                                                            \
	TEST_CASE(fmt::format("Test ArrayView -- {} {}", STRINGIFY(SCALAR), STRINGIFY(BACKEND)),        \
			  "[array-lib]") {                                                                     \
		lrc::Shape shape({7, 11});                                                                 \
		lrc::Array<SCALAR, BACKEND> testArr(shape);                                                 \
                                                                                                   \
		for (int64_t i = 0; i < testArr.shape().size(); ++i) { testArr.storage()[i] = i; }         \
                                                                                                   \
		auto testView		  = lrc::array::ArrayView(testArr);                                    \
		auto testViewCopy	  = lrc::array::ArrayView(testView);                                   \
		auto testViewMoveView = lrc::array::ArrayView(lrc::array::ArrayView(testArr));             \
                                                                                                   \
		REQUIRE(testView.ndim() == 2);                                                             \
		REQUIRE(testViewCopy.ndim() == 2);                                                         \
		REQUIRE(testViewMoveView.ndim() == 2);                                                     \
                                                                                                   \
		REQUIRE(testView.shape() == shape);                                                        \
		REQUIRE(testViewCopy.shape() == shape);                                                    \
		REQUIRE(testViewMoveView.shape() == shape);                                                \
                                                                                                   \
		auto checkValues = [](const auto &view) {                                                  \
			if (view.ndim() == 2) {                                                                \
				for (int64_t row = 0; row < view.shape()[0]; ++row) {                              \
					for (int64_t col = 0; col < view.shape()[1]; ++col) {                          \
						REQUIRE(view[row][col].get() == row * view.shape()[1] + col);              \
					}                                                                              \
				}                                                                                  \
			} else if (view.ndim() == 3) {                                                         \
				for (int64_t row = 0; row < view.shape()[0]; ++row) {                              \
					for (int64_t col = 0; col < view.shape()[1]; ++col) {                          \
						for (int64_t depth = 0; depth < view.shape()[2]; ++depth) {                \
							REQUIRE(view[row][col][depth].get() ==                                 \
									row * view.shape()[1] * view.shape()[2] +                      \
									  col * view.shape()[2] + depth);                              \
						}                                                                          \
					}                                                                              \
				}                                                                                  \
			} else {                                                                               \
				REQUIRE(true);                                                                     \
			}                                                                                      \
		};                                                                                         \
                                                                                                   \
		checkValues(testView);                                                                     \
		checkValues(testViewCopy);                                                                 \
		checkValues(testViewMoveView);                                                             \
                                                                                                   \
		auto evalTest		  = testView.eval();                                                   \
		auto evalTestCopy	  = testViewCopy.eval();                                               \
		auto evalTestMoveView = testViewMoveView.eval();                                           \
                                                                                                   \
		REQUIRE(evalTest.ndim() == 2);                                                             \
		REQUIRE(evalTestCopy.ndim() == 2);                                                         \
		REQUIRE(evalTestMoveView.ndim() == 2);                                                     \
                                                                                                   \
		REQUIRE(evalTest.shape() == shape);                                                        \
		REQUIRE(evalTestCopy.shape() == shape);                                                    \
		REQUIRE(evalTestMoveView.shape() == shape);                                                \
                                                                                                   \
		for (int64_t i = 0; i < evalTest.shape().size(); ++i) {                                    \
			REQUIRE(evalTest.storage()[i] == i);                                                   \
			REQUIRE(evalTestCopy.storage()[i] == i);                                               \
			REQUIRE(evalTestMoveView.storage()[i] == i);                                           \
		}                                                                                          \
	}

// TEST_ARRAY_VIEW(int8_t, lrc::backend::CPU)
TEST_ARRAY_VIEW(int16_t, lrc::backend::CPU)
TEST_ARRAY_VIEW(int32_t, lrc::backend::CPU)
TEST_ARRAY_VIEW(int64_t, lrc::backend::CPU)
TEST_ARRAY_VIEW(float, lrc::backend::CPU)
TEST_ARRAY_VIEW(double, lrc::backend::CPU)

#if defined(LIBRAPID_HAS_OPENCL)
TEST_CASE("Configure OpenCL", "[array-lib]") {
	lrc::configureOpenCL();
}

// TEST_ARRAY_VIEW(int8_t, lrc::backend::OpenCL)
TEST_ARRAY_VIEW(int16_t, lrc::backend::OpenCL)
TEST_ARRAY_VIEW(int32_t, lrc::backend::OpenCL)
TEST_ARRAY_VIEW(int64_t, lrc::backend::OpenCL)
TEST_ARRAY_VIEW(float, lrc::backend::OpenCL)
TEST_ARRAY_VIEW(double, lrc::backend::OpenCL)
#endif

#if defined(LIBRAPID_HAS_CUDA)
// TEST_ARRAY_VIEW(int8_t, lrc::backend::CUDA)
TEST_ARRAY_VIEW(int16_t, lrc::backend::CUDA)
TEST_ARRAY_VIEW(int32_t, lrc::backend::CUDA)
TEST_ARRAY_VIEW(int64_t, lrc::backend::CUDA)
TEST_ARRAY_VIEW(float, lrc::backend::CUDA)
TEST_ARRAY_VIEW(double, lrc::backend::CUDA)
#endif
