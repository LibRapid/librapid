#ifndef LIBRAPID_ARRAY_LINALG
#define LIBRAPID_ARRAY_LINALG

namespace librapid::typetraits {
    template<typename T>
    struct IsBlasType : std::false_type {};

    template<>
    struct IsBlasType<half> : std::true_type {};

    template<>
    struct IsBlasType<float> : std::true_type {};

    template<>
    struct IsBlasType<double> : std::true_type {};

    template<>
    struct IsBlasType<Complex<float>> : std::true_type {};

    template<>
    struct IsBlasType<Complex<double>> : std::true_type {};
} // namespace librapid::typetraits

#include "transpose.hpp"

#include "level2/gemv.hpp"

#include "level3/geam.hpp"
#include "level3/gemm.hpp"

#include "arrayMultiply.hpp"

#include "compat.hpp"

#endif // LIBRAPID_ARRAY_LINALG