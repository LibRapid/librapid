#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc           = librapid;
static double tolerance = 1e-5;

// using SCALAR = double;

#define TEST_COMPLEX(SCALAR)                                                                       \
    TEST_CASE(fmt::format("Test Complex {}", STRINGIFY(SCALAR)), "[math]") {                       \
        SECTION("Constructors") {                                                                  \
            lrc::Complex<SCALAR> z1;                                                               \
            REQUIRE(z1.real() == 0);                                                               \
            REQUIRE(z1.imag() == 0);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z2(1, 2);                                                         \
            REQUIRE(z2.real() == 1);                                                               \
            REQUIRE(z2.imag() == 2);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z3(z2);                                                           \
            REQUIRE(z3.real() == 1);                                                               \
            REQUIRE(z3.imag() == 2);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z4 = z2;                                                          \
            REQUIRE(z4.real() == 1);                                                               \
            REQUIRE(z4.imag() == 2);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z5 = {1, 2};                                                      \
            REQUIRE(z5.real() == 1);                                                               \
            REQUIRE(z5.imag() == 2);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z6(1);                                                            \
            REQUIRE(z6.real() == 1);                                                               \
            REQUIRE(z6.imag() == 0);                                                               \
                                                                                                   \
            lrc::Complex<SCALAR> z7(lrc::Complex<SCALAR>(1, 2));                                   \
            REQUIRE(z7.real() == 1);                                                               \
            REQUIRE(z7.imag() == 2);                                                               \
                                                                                                   \
            z7 = 123;                                                                              \
            REQUIRE(z7.real() == 123);                                                             \
            REQUIRE(z7.imag() == 0);                                                               \
                                                                                                   \
            z1.real(5);                                                                            \
            z1.imag(10);                                                                           \
            REQUIRE(z1.real() == 5);                                                               \
            REQUIRE(z1.imag() == 10);                                                              \
                                                                                                   \
            REQUIRE(lrc::real(z1) == z1.real());                                                   \
            REQUIRE(lrc::imag(z1) == z1.imag());                                                   \
        }                                                                                          \
                                                                                                   \
        SECTION("Inplace Arithmetic") {                                                            \
            lrc::Complex<SCALAR> z1(1, 2);                                                         \
            lrc::Complex<SCALAR> z2(3, 4);                                                         \
                                                                                                   \
            z1 += SCALAR(1);                                                                       \
            REQUIRE(z1.real() == 2);                                                               \
            REQUIRE(z1.imag() == 2);                                                               \
                                                                                                   \
            z1 -= SCALAR(1);                                                                       \
            REQUIRE(z1.real() == 1);                                                               \
            REQUIRE(z1.imag() == 2);                                                               \
                                                                                                   \
            z1 *= SCALAR(2);                                                                       \
            REQUIRE(z1.real() == 2);                                                               \
            REQUIRE(z1.imag() == 4);                                                               \
                                                                                                   \
            z1 /= SCALAR(2);                                                                       \
            REQUIRE(z1.real() == 1);                                                               \
            REQUIRE(z1.imag() == 2);                                                               \
                                                                                                   \
            z1 += z2;                                                                              \
            REQUIRE(z1.real() == 4);                                                               \
            REQUIRE(z1.imag() == 6);                                                               \
                                                                                                   \
            z1 -= z2;                                                                              \
            REQUIRE(z1.real() == 1);                                                               \
            REQUIRE(z1.imag() == 2);                                                               \
                                                                                                   \
            z1 *= z2;                                                                              \
            REQUIRE(z1.real() == -5);                                                              \
            REQUIRE(z1.imag() == 10);                                                              \
                                                                                                   \
            z1 /= z2;                                                                              \
            REQUIRE(z1.real() == 1);                                                               \
            REQUIRE(z1.imag() == 2);                                                               \
        }                                                                                          \
                                                                                                   \
        SECTION("Casting") {                                                                       \
            lrc::Complex<SCALAR> z1(1, 2);                                                         \
            lrc::Complex<SCALAR> z2(3, 4);                                                         \
                                                                                                   \
            REQUIRE((int)z1 == 1);                                                                 \
            REQUIRE((int)z2 == 3);                                                                 \
                                                                                                   \
            REQUIRE(lrc::Complex<int>(z1) == lrc::Complex<int>(1, 2));                             \
            REQUIRE(lrc::Complex<int>(z2) == lrc::Complex<int>(3, 4));                             \
                                                                                                   \
            REQUIRE(z1.str() == fmt::format("({}+{}j)", z1.real(), z1.imag()));                    \
            REQUIRE(z2.str() == fmt::format("({}+{}j)", z2.real(), z2.imag()));                    \
            REQUIRE((-z1).str() == fmt::format("(-{}-{}j)", z1.real(), z1.imag()));                \
            REQUIRE((-z2).str() == fmt::format("(-{}-{}j)", z2.real(), z2.imag()));                \
        }                                                                                          \
                                                                                                   \
        SECTION("Out-of-Place Arithmetic") {                                                       \
            lrc::Complex<SCALAR> z1(1, 2);                                                         \
            lrc::Complex<SCALAR> z2(3, 4);                                                         \
                                                                                                   \
            auto neg = -z1;                                                                        \
            REQUIRE(neg.real() == -1);                                                             \
            REQUIRE(neg.imag() == -2);                                                             \
                                                                                                   \
            auto add1 = z1 + z2;                                                                   \
            REQUIRE(add1.real() == 4);                                                             \
            REQUIRE(add1.imag() == 6);                                                             \
                                                                                                   \
            auto sub1 = z1 - z2;                                                                   \
            REQUIRE(sub1.real() == -2);                                                            \
            REQUIRE(sub1.imag() == -2);                                                            \
                                                                                                   \
            auto mul1 = z1 * z2;                                                                   \
            REQUIRE(mul1.real() == -5);                                                            \
            REQUIRE(mul1.imag() == 10);                                                            \
                                                                                                   \
            auto div1 = z1 / z2;                                                                   \
            REQUIRE(lrc::isClose(div1.real(), 0.44, tolerance));                                   \
            REQUIRE(lrc::isClose(div1.imag(), 0.08, tolerance));                                   \
                                                                                                   \
            auto add2 = z1 + 1;                                                                    \
            REQUIRE(add2.real() == 2);                                                             \
            REQUIRE(add2.imag() == 2);                                                             \
                                                                                                   \
            auto sub2 = z1 - 1;                                                                    \
            REQUIRE(sub2.real() == 0);                                                             \
            REQUIRE(sub2.imag() == 2);                                                             \
                                                                                                   \
            auto mul2 = z1 * 2;                                                                    \
            REQUIRE(mul2.real() == 2);                                                             \
            REQUIRE(mul2.imag() == 4);                                                             \
                                                                                                   \
            auto div2 = z1 / 2;                                                                    \
            REQUIRE(lrc::isClose(div2.real(), 0.5, tolerance));                                    \
            REQUIRE(lrc::isClose(div2.imag(), 1.0, tolerance));                                    \
                                                                                                   \
            auto add3 = 1 + z1;                                                                    \
            REQUIRE(add3.real() == 2);                                                             \
            REQUIRE(add3.imag() == 2);                                                             \
                                                                                                   \
            auto sub3 = 1 - z1;                                                                    \
            REQUIRE(sub3.real() == 0);                                                             \
            REQUIRE(sub3.imag() == -2);                                                            \
                                                                                                   \
            auto mul3 = 2 * z1;                                                                    \
            REQUIRE(mul3.real() == 2);                                                             \
            REQUIRE(mul3.imag() == 4);                                                             \
                                                                                                   \
            auto div3 = 2 / z1;                                                                    \
            REQUIRE(lrc::isClose(div3.real(), 0.4, tolerance));                                    \
            REQUIRE(lrc::isClose(div3.imag(), -0.8, tolerance));                                   \
        }                                                                                          \
                                                                                                   \
        SECTION("Complex Functions") {                                                             \
            lrc::Complex<SCALAR> z1(1, 2);                                                         \
            lrc::Complex<SCALAR> z2(-3, 4);                                                        \
                                                                                                   \
            REQUIRE(lrc::sqrt(z2) == lrc::Complex<SCALAR>(1, 2));                                  \
            REQUIRE(lrc::isClose(lrc::abs(z1), lrc::sqrt(SCALAR(5)), tolerance));                  \
            REQUIRE(lrc::isClose(lrc::abs(z2), 5, tolerance));                                     \
            REQUIRE(lrc::conj(z1) == lrc::Complex<SCALAR>(1, -2));                                 \
            REQUIRE(lrc::conj(z2) == lrc::Complex<SCALAR>(-3, -4));                                \
                                                                                                   \
            auto acos = lrc::acos(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(acos.real(), 1.143717740402420493750674808320794582795, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(acos.imag(), -1.528570919480998161272456184793673393288, tolerance));   \
                                                                                                   \
            auto acosh = lrc::acosh(z1);                                                           \
            REQUIRE(                                                                               \
              lrc::isClose(acosh.real(), 1.528570919480998161272456184793673393288, tolerance));   \
            REQUIRE(                                                                               \
              lrc::isClose(acosh.imag(), 1.143717740402420493750674808320794582795, tolerance));   \
                                                                                                   \
            auto asinh = lrc::asinh(z1);                                                           \
            REQUIRE(                                                                               \
              lrc::isClose(asinh.real(), 1.469351744368185273255844317361647616787, tolerance));   \
            REQUIRE(                                                                               \
              lrc::isClose(asinh.imag(), 1.063440023577752056189491997089551002851, tolerance));   \
                                                                                                   \
            auto asin = lrc::asin(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(asin.real(), 0.42707858639247612548064688331895685930333, tolerance));  \
            REQUIRE(                                                                               \
              lrc::isClose(asin.imag(), 1.52857091948099816127245618479367339328868, tolerance));  \
                                                                                                   \
            auto atanh = lrc::atanh(z1);                                                           \
            REQUIRE(                                                                               \
              lrc::isClose(atanh.real(), 0.173286795139986351536318642871984660455, tolerance));   \
            REQUIRE(                                                                               \
              lrc::isClose(atanh.imag(), 1.178097245096172464423491268729813577364, tolerance));   \
                                                                                                   \
            auto atan = lrc::atan(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(atan.real(), 1.338972522294493561202819911642758892643, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(atan.imag(), 0.402359478108525093650936383865827688755, tolerance));    \
                                                                                                   \
            auto cosh = lrc::cosh(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(cosh.real(), -0.64214812471551996484480068696227878947, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(cosh.imag(), 1.068607421382778339597440033783951588665, tolerance));    \
                                                                                                   \
            auto exp = lrc::exp(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(exp.real(), -1.131204383756813638431255255510794710628, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(exp.imag(), 2.4717266720048189276169308935516645327361, tolerance));    \
                                                                                                   \
            auto exp2 = lrc::exp2(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(exp2.real(), 0.366913949486603353679882473618470209036, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(exp2.imag(), 1.966055480822487441172329700685456305222, tolerance));    \
                                                                                                   \
            auto exp10 = lrc::exp10(z1);                                                           \
            REQUIRE(                                                                               \
              lrc::isClose(exp10.real(), -1.0701348355877020772086517528518239460, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(exp10.imag(), -9.9425756941378968736161937190915602112, tolerance));    \
                                                                                                   \
            auto log = lrc::log(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(log.real(), 0.804718956217050314461503047313945610162, tolerance));     \
            REQUIRE(                                                                               \
              lrc::isClose(log.imag(), 1.107148717794090503017065460178537040070, tolerance));     \
                                                                                                   \
            auto log2 = lrc::log2(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(log2.real(), 1.1609640474436811739351597147446950879, tolerance));      \
            REQUIRE(                                                                               \
              lrc::isClose(log2.imag(), 1.5972779646881088066382317418569791182, tolerance));      \
                                                                                                   \
            auto pow3 = lrc::pow(z1, 3);                                                           \
            REQUIRE(lrc::isClose(pow3.real(), -11, tolerance));                                    \
            REQUIRE(lrc::isClose(pow3.imag(), -2, tolerance));                                     \
                                                                                                   \
            auto realPow = lrc::pow(SCALAR(5), z1);                                                \
            REQUIRE(                                                                               \
              lrc::isClose(realPow.real(), -4.98507570899023509256310961483534856535, tolerance)); \
            REQUIRE(                                                                               \
              lrc::isClose(realPow.imag(), -0.38603131431984235432537596434762968808, tolerance)); \
                                                                                                   \
            auto sinh = lrc::sinh(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(sinh.real(), -0.48905625904129372065865106274460904854, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(sinh.imag(), 1.4031192506220405511576005806225837627, tolerance));      \
                                                                                                   \
            auto sqrt = lrc::sqrt(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(sqrt.real(), 1.272019649514068964252422461737491491715, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(sqrt.imag(), 0.786151377757423286069558585842958929523, tolerance));    \
                                                                                                   \
            auto tanh = lrc::tanh(z1);                                                             \
            REQUIRE(                                                                               \
              lrc::isClose(tanh.real(), 1.166736257240919881810070397144984248593, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(tanh.imag(), -0.24345820118572525270261038865215160145, tolerance));    \
                                                                                                   \
            auto arg = lrc::arg(z1);                                                               \
            REQUIRE(lrc::isClose(arg, lrc::atan(2), tolerance));                                   \
                                                                                                   \
            auto cos = lrc::cos(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(cos.real(), 2.0327230070196655294363434484995142637319, tolerance));    \
            REQUIRE(                                                                               \
              lrc::isClose(cos.imag(), -3.051897799151800057512115686895105452888, tolerance));    \
                                                                                                   \
            auto csc = lrc::csc(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(csc.real(), 0.22837506559968659341093330251058291161553, tolerance));   \
            REQUIRE(                                                                               \
              lrc::isClose(csc.imag(), -0.1413630216124078007231203906301757072451, tolerance));   \
                                                                                                   \
            auto sec = lrc::sec(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(sec.real(), 0.15117629826557722714368596016961254310795, tolerance));   \
            REQUIRE(                                                                               \
              lrc::isClose(sec.imag(), 0.22697367539372159536972826811917694791070, tolerance));   \
                                                                                                   \
            auto cot = lrc::cot(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(cot.real(), 0.032797755533752594062764546576583062934, tolerance));     \
            REQUIRE(                                                                               \
              lrc::isClose(cot.imag(), -0.984329226458191029471888181689464448193, tolerance));    \
                                                                                                   \
            auto acsc = lrc::acsc(lrc::csc(z1));                                                   \
            REQUIRE(lrc::isClose(acsc.real(), z1.real(), tolerance));                              \
            REQUIRE(lrc::isClose(acsc.imag(), z1.imag(), tolerance));                              \
                                                                                                   \
            auto asec = lrc::asec(lrc::sec(z1));                                                   \
            REQUIRE(lrc::isClose(asec.real(), z1.real(), tolerance));                              \
            REQUIRE(lrc::isClose(asec.imag(), z1.imag(), tolerance));                              \
                                                                                                   \
            auto acot = lrc::acot(lrc::cot(z1));                                                   \
            REQUIRE(lrc::isClose(acot.real(), z1.real(), tolerance));                              \
            REQUIRE(lrc::isClose(acot.imag(), z1.imag(), tolerance));                              \
                                                                                                   \
            REQUIRE(lrc::norm(z1) == 5);                                                           \
                                                                                                   \
            auto polar = lrc::polar(lrc::sqrt(SCALAR(5)), lrc::atan(SCALAR(2)));                   \
            REQUIRE(lrc::isClose(polar.real(), 1, tolerance));                                     \
            REQUIRE(lrc::isClose(polar.imag(), 2, tolerance));                                     \
                                                                                                   \
            auto sin = lrc::sin(z1);                                                               \
            REQUIRE(                                                                               \
              lrc::isClose(sin.real(), 3.165778513216168146740734617191905538379110, tolerance));  \
            REQUIRE(                                                                               \
              lrc::isClose(sin.imag(), 1.959601041421605897070352049989358278436320, tolerance));  \
                                                                                                   \
            auto floor = lrc::floor(lrc::Complex<SCALAR>(1.5, 2.5));                               \
            REQUIRE(floor == lrc::Complex<SCALAR>(1, 2));                                          \
                                                                                                   \
            auto ceil = lrc::ceil(lrc::Complex<SCALAR>(1.5, 2.5));                                 \
            REQUIRE(ceil == lrc::Complex<SCALAR>(2, 3));                                           \
        }                                                                                          \
    }

TEST_COMPLEX(float)
TEST_COMPLEX(double)

#if defined(LIBRAPID_USE_MULTIPREC)
TEST_COMPLEX(lrc::mpfr)
#endif // LIBRAPID_USE_MULTIPREC
