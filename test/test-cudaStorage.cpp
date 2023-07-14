#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

#if defined(LIBRAPID_HAS_CUDA)

#	define REGISTER_CASES(TYPE)                                                                   \
		SECTION("Type: " STRINGIFY(TYPE)) {                                                        \
			using ScalarType = TYPE;                                                               \
			lrc::CudaStorage<ScalarType> storage(5);                                               \
                                                                                                   \
			REQUIRE(storage.size() == 5);                                                          \
                                                                                                   \
			storage[0] = 1;                                                                        \
			storage[1] = 10;                                                                       \
                                                                                                   \
			REQUIRE(storage[0] == 1);                                                              \
			REQUIRE(storage[1] == 10);                                                             \
                                                                                                   \
			lrc::CudaStorage<ScalarType> storage2({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});                \
                                                                                                   \
			REQUIRE(storage2.size() == 10);                                                        \
			REQUIRE(storage2[0] == 1);                                                             \
			REQUIRE(storage2[1] == 2);                                                             \
			REQUIRE(storage2[8] == 9);                                                             \
			REQUIRE(storage2[9] == 10);                                                            \
                                                                                                   \
			lrc::CudaStorage<ScalarType> storage3(10, 1);                                          \
                                                                                                   \
			REQUIRE(storage3.size() == 10);                                                        \
			REQUIRE(storage3[0] == 1);                                                             \
			REQUIRE(storage3[1] == 1);                                                             \
			REQUIRE(storage3[8] == 1);                                                             \
			REQUIRE(storage3[9] == 1);                                                             \
                                                                                                   \
			auto storage4 = lrc::CudaStorage<ScalarType>(storage2);                                \
                                                                                                   \
			REQUIRE(storage4.size() == 10);                                                        \
			REQUIRE(storage4[0] == 1);                                                             \
			REQUIRE(storage4[1] == 2);                                                             \
			REQUIRE(storage4[8] == 9);                                                             \
			REQUIRE(storage4[9] == 10);                                                            \
                                                                                                   \
			/* storage4 = lrc::CudaStorage<ScalarType>(100);   */                                  \
			/* REQUIRE(storage4.size() == 100);                */                                  \
			/* storage4[0]	 = 1;                               */                                  \
			/* storage4[1]	 = 2;                               */                                  \
			/* storage4[98] = 99;                              */                                  \
			/* storage4[99] = 100;                             */                                  \
			/* REQUIRE(storage4[0] == 1);                      */                                  \
			/* REQUIRE(storage4[1] == 2);                      */                                  \
			/* REQUIRE(storage4[98] == 99);                    */                                  \
			/* REQUIRE(storage4[99] == 100);                   */                                  \
                                                                                                   \
			storage4 = storage3;                                                                   \
                                                                                                   \
			REQUIRE(storage4.size() == 10);                                                        \
			REQUIRE(storage4[0] == 1);                                                             \
			REQUIRE(storage4[1] == 1);                                                             \
			REQUIRE(storage4[8] == 1);                                                             \
			REQUIRE(storage4[9] == 1);                                                             \
                                                                                                   \
			lrc::CudaStorage<ScalarType> storage6(20, 123);                                        \
			REQUIRE(storage6.size() == 20);                                                        \
			storage6.resize(5);                                                                    \
			REQUIRE(storage6.size() == 5);                                                         \
			REQUIRE(storage6[0] == 123);                                                           \
			REQUIRE(storage6[1] == 123);                                                           \
			REQUIRE(storage6[2] == 123);                                                           \
			REQUIRE(storage6[3] == 123);                                                           \
			REQUIRE(storage6[4] == 123);                                                           \
                                                                                                   \
			storage6.resize(10);                                                                   \
			REQUIRE(storage6.size() == 10);                                                        \
			REQUIRE(storage6[0] == 123);                                                           \
			REQUIRE(storage6[1] == 123);                                                           \
			REQUIRE(storage6[2] == 123);                                                           \
			REQUIRE(storage6[3] == 123);                                                           \
			REQUIRE(storage6[4] == 123);                                                           \
                                                                                                   \
			storage6.resize(100, 0);                                                               \
			REQUIRE(storage6.size() == 100);                                                       \
		}

#	define BENCHMARK_CONSTRUCTORS(TYPE_, FILL_)                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 10") {                                        \
			lrc::CudaStorage<TYPE_> storage(10);                                                   \
			return storage.size();                                                                 \
		};                                                                                         \
                                                                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 1000") {                                      \
			lrc::CudaStorage<TYPE_> storage(1000);                                                 \
			return storage.size();                                                                 \
		};                                                                                         \
                                                                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 1000000") {                                   \
			lrc::CudaStorage<TYPE_> storage(1000000);                                              \
			return storage.size();                                                                 \
		};                                                                                         \
                                                                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 10 FILLED") {                                 \
			lrc::CudaStorage<TYPE_> storage(10, FILL_);                                            \
			return storage.size();                                                                 \
		};                                                                                         \
                                                                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 1000 FILLED") {                               \
			lrc::CudaStorage<TYPE_> storage(1000, FILL_);                                          \
			return storage.size();                                                                 \
		};                                                                                         \
                                                                                                   \
		BENCHMARK("CudaStorage<" STRINGIFY(TYPE_) "> 1000000 FILLED") {                            \
			lrc::CudaStorage<TYPE_> storage(1000000, FILL_);                                       \
			return storage.size();                                                                 \
		}

TEST_CASE("Test CudaStorage<T>", "[storage]") {
	SECTION("Test CudaStorage") {
		REGISTER_CASES(char);
		REGISTER_CASES(unsigned char);
		REGISTER_CASES(short);
		REGISTER_CASES(unsigned short);
		REGISTER_CASES(int);
		REGISTER_CASES(unsigned int);
		REGISTER_CASES(long);
		REGISTER_CASES(unsigned long);
		REGISTER_CASES(long long);
		REGISTER_CASES(unsigned long long);
		REGISTER_CASES(float);
		REGISTER_CASES(double);
		REGISTER_CASES(long double);
	}

	SECTION("Benchmarks") {
		BENCHMARK_CONSTRUCTORS(int, 123);
		BENCHMARK_CONSTRUCTORS(double, 456);
	}
}

#else

TEST_CASE("Default", "[storage]") {
	LIBRAPID_WARN("OpenCL not available, skipping tests");
	SECTION("Default") { REQUIRE(true); }
}

#endif // LIBRAPID_HAS_CUDA
