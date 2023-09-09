#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

#define REGISTER_CASES(TYPE)                                                                       \
    SECTION("Type: " STRINGIFY(TYPE)) {                                                            \
        using ScalarType = TYPE;                                                                   \
        lrc::Storage<ScalarType> storage(5);                                                       \
                                                                                                   \
        REQUIRE(storage.size() == 5);                                                              \
                                                                                                   \
        storage[0] = 1;                                                                            \
        storage[1] = 10;                                                                           \
                                                                                                   \
        REQUIRE(storage[0] == 1);                                                                  \
        REQUIRE(storage[1] == 10);                                                                 \
                                                                                                   \
        lrc::Storage<ScalarType> storage2({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});                        \
                                                                                                   \
        REQUIRE(storage2.size() == 10);                                                            \
        REQUIRE(storage2[0] == 1);                                                                 \
        REQUIRE(storage2[1] == 2);                                                                 \
        REQUIRE(storage2[8] == 9);                                                                 \
        REQUIRE(storage2[9] == 10);                                                                \
                                                                                                   \
        lrc::Storage<ScalarType> storage3(100, 1);                                                 \
                                                                                                   \
        REQUIRE(storage3.size() == 100);                                                           \
        REQUIRE(storage3[0] == 1);                                                                 \
        REQUIRE(storage3[1] == 1);                                                                 \
        REQUIRE(storage3[98] == 1);                                                                \
        REQUIRE(storage3[99] == 1);                                                                \
                                                                                                   \
        auto storage4 = lrc::Storage<ScalarType>(storage2);                                        \
                                                                                                   \
        REQUIRE(storage4.size() == 10);                                                            \
        REQUIRE(storage4[0] == 1);                                                                 \
        REQUIRE(storage4[1] == 2);                                                                 \
        REQUIRE(storage4[8] == 9);                                                                 \
        REQUIRE(storage4[9] == 10);                                                                \
                                                                                                   \
        /* storage4 = lrc::Storage<ScalarType>(100);     */                                        \
        /* REQUIRE(storage4.size() == 100);              */                                        \
        /* storage4[0]	 = 1;                            */                                         \
        /* storage4[1]	 = 2;                            */                                         \
        /* storage4[98] = 99;                            */                                        \
        /* storage4[99] = 100;                           */                                        \
        /* REQUIRE(storage4[0] == 1);                    */                                        \
        /* REQUIRE(storage4[1] == 2);                    */                                        \
        /* REQUIRE(storage4[98] == 99);                  */                                        \
        /* REQUIRE(storage4[99] == 100);                 */                                        \
                                                                                                   \
        storage4 = storage3;                                                                       \
                                                                                                   \
        REQUIRE(storage4.size() == 100);                                                           \
        REQUIRE(storage4[0] == 1);                                                                 \
        REQUIRE(storage4[1] == 1);                                                                 \
        REQUIRE(storage4[98] == 1);                                                                \
        REQUIRE(storage4[99] == 1);                                                                \
                                                                                                   \
        SECTION("Const Iterator") {                                                                \
            ScalarType i = 1;                                                                      \
            for (const auto &val : storage2) {                                                     \
                REQUIRE(val == i);                                                                 \
                i += 1;                                                                            \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        SECTION("Non-Const Iterator") {                                                            \
            ScalarType i = 1;                                                                      \
            for (auto &val : storage2) {                                                           \
                REQUIRE(val == i);                                                                 \
                i += 1;                                                                            \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        lrc::Storage<ScalarType> storage6(20, 123);                                                \
        REQUIRE(storage6.size() == 20);                                                            \
        storage6.resize(5);                                                                        \
        REQUIRE(storage6.size() == 5);                                                             \
        REQUIRE(storage6[0] == 123);                                                               \
        REQUIRE(storage6[1] == 123);                                                               \
        REQUIRE(storage6[2] == 123);                                                               \
        REQUIRE(storage6[3] == 123);                                                               \
        REQUIRE(storage6[4] == 123);                                                               \
                                                                                                   \
        storage6.resize(10);                                                                       \
        REQUIRE(storage6.size() == 10);                                                            \
        REQUIRE(storage6[0] == 123);                                                               \
        REQUIRE(storage6[1] == 123);                                                               \
        REQUIRE(storage6[2] == 123);                                                               \
        REQUIRE(storage6[3] == 123);                                                               \
        REQUIRE(storage6[4] == 123);                                                               \
                                                                                                   \
        storage6.resize(100, 0);                                                                   \
        REQUIRE(storage6.size() == 100);                                                           \
    }

#define BENCHMARK_CONSTRUCTORS(TYPE_, FILL_)                                                       \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 10") {                                                \
        lrc::Storage<TYPE_> storage(10);                                                           \
        return storage.size();                                                                     \
    };                                                                                             \
                                                                                                   \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 1000") {                                              \
        lrc::Storage<TYPE_> storage(1000);                                                         \
        return storage.size();                                                                     \
    };                                                                                             \
                                                                                                   \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 1000000") {                                           \
        lrc::Storage<TYPE_> storage(1000000);                                                      \
        return storage.size();                                                                     \
    };                                                                                             \
                                                                                                   \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 10 FILLED") {                                         \
        lrc::Storage<TYPE_> storage(10, FILL_);                                                    \
        return storage.size();                                                                     \
    };                                                                                             \
                                                                                                   \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 1000 FILLED") {                                       \
        lrc::Storage<TYPE_> storage(1000, FILL_);                                                  \
        return storage.size();                                                                     \
    };                                                                                             \
                                                                                                   \
    BENCHMARK("Storage<" STRINGIFY(TYPE_) "> 1000000 FILLED") {                                    \
        lrc::Storage<TYPE_> storage(1000000, FILL_);                                               \
        return storage.size();                                                                     \
    }

TEST_CASE("Test Storage<T>", "[storage]") {
    SECTION("Trivially Constructible Storage") {
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

    SECTION("Non-Trivially Constructible Storage") {
        // Can't use normal tests, so just test a few things
        lrc::Storage<std::string> storage(5);
        REQUIRE(storage.size() == 5);
        storage[0] = "Hello";
        storage[1] = "World";
        REQUIRE(storage[0] == "Hello");
        REQUIRE(storage[1] == "World");

        lrc::Storage<std::string> storage2({"Hello", "World"});
        REQUIRE(storage2.size() == 2);
        REQUIRE(storage2[0] == "Hello");
        REQUIRE(storage2[1] == "World");

        lrc::Storage<std::string> storage3(20, "Hello");
        REQUIRE(storage3.size() == 20);
        REQUIRE(storage3[0] == "Hello");
        REQUIRE(storage3[1] == "Hello");
        REQUIRE(storage3[18] == "Hello");
        REQUIRE(storage3[19] == "Hello");

        auto storage4 = lrc::Storage<std::vector<std::string>>(10);
        REQUIRE(storage4.size() == 10);
        storage4[0].push_back("Hello");
        storage4[0].push_back("World");
        REQUIRE(storage4[0][0] == "Hello");
        REQUIRE(storage4[0][1] == "World");

        struct Three {
            int a;
            int b;
            int c;
        };

        auto storage5 = lrc::Storage<Three>(10);
        REQUIRE(storage5.size() == 10);
        storage5[0].a = 1;
        storage5[0].b = 2;
        storage5[0].c = 3;
        storage5[1].a = 4;
        storage5[1].b = 5;
        storage5[1].c = 6;
        REQUIRE(storage5[0].a == 1);
        REQUIRE(storage5[0].b == 2);
        REQUIRE(storage5[0].c == 3);
        REQUIRE(storage5[1].a == 4);
        REQUIRE(storage5[1].b == 5);
        REQUIRE(storage5[1].c == 6);

        auto storage6 = storage5;
        REQUIRE(storage5[0].a == 1);
        REQUIRE(storage5[0].b == 2);
        REQUIRE(storage5[0].c == 3);
        REQUIRE(storage5[1].a == 4);
        REQUIRE(storage5[1].b == 5);
        REQUIRE(storage5[1].c == 6);
    }

    SECTION("Benchmarks") {
        BENCHMARK_CONSTRUCTORS(int, 123);
        BENCHMARK_CONSTRUCTORS(double, 456);
        BENCHMARK_CONSTRUCTORS(std::string, "Hello, World");
        BENCHMARK_CONSTRUCTORS(std::vector<int>, {1 COMMA 2 COMMA 3 COMMA 4});
        BENCHMARK_CONSTRUCTORS(std::vector<double>, {1 COMMA 2 COMMA 3 COMMA 4});
    }
}
