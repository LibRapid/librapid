INCLUDE(CheckCXXSourceRuns)

set(COMPILER_GNU false)
set(COMPILER_INTEL false)
set(COMPILER_CLANG false)
set(COMPILER_MSVC false)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(COMPILER_GNU true)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(COMPILER_INTEL true)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMPILER_CLANG true)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(COMPILER_MSVC true)
else ()
    # Unknown Compiler
endif ()

set(LIBRAPID_ARCH_FLAGS)
set(LIBRAPID_ARCH_FOUND)

# Function to test a given SIMD capability
function(check_simd_capability FLAG_GNU FLAG_MSVC NAME TEST_SOURCE VAR)
    set(CMAKE_REQUIRED_FLAGS)
    if (COMPILER_GNU OR COMPILER_INTEL OR COMPILER_CLANG)
        set(CMAKE_REQUIRED_FLAGS "${FLAG_GNU}")
    elseif (COMPILER_MSVC)  # reserve for WINDOWS
        set(CMAKE_REQUIRED_FLAGS "${FLAG_MSVC}")
    endif ()

    CHECK_CXX_SOURCE_RUNS("${TEST_SOURCE}" ${VAR})

    if (${${VAR}})
        if (COMPILER_GNU OR COMPILER_INTEL OR COMPILER_CLANG)
            # set(LIBRAPID_ARCH_FLAGS "${LIBRAPID_ARCH_FLAGS} ${FLAG_GNU}" PARENT_SCOPE)

            list(APPEND LIBRAPID_ARCH_FLAGS ${FLAG_GNU})
            set(LIBRAPID_ARCH_FLAGS ${LIBRAPID_ARCH_FLAGS} PARENT_SCOPE)

            message(STATUS "[ LIBRAPID ] ${NAME} found: ${FLAG_GNU}")
        elseif (MSVC)
            # set(LIBRAPID_ARCH_FLAGS "${LIBRAPID_ARCH_FLAGS} ${FLAG_MSVC}" PARENT_SCOPE)

            list(APPEND LIBRAPID_ARCH_FLAGS ${FLAG_MSVC})
            set(LIBRAPID_ARCH_FLAGS ${LIBRAPID_ARCH_FLAGS} PARENT_SCOPE)

            message(STATUS "[ LIBRAPID ] ${NAME} found: ${FLAG_MSVC}")
        endif ()
        set(LIBRAPID_ARCH_FOUND TRUE PARENT_SCOPE)
    else ()
        message(STATUS "[ LIBRAPID ] ${NAME} not found")
    endif ()
endfunction()

check_simd_capability("-mmmx" "" "MMX" "
#include <mmintrin.h>
int main() {
    __m64 a = _mm_set_pi32(-1, 2);
    __m64 result = _mm_abs_pi32(a);
    return 0;
}" SIMD_MMX)

# Check SSE2 (not a valid flag for MSVC)
check_simd_capability("-msse2" "" "SSE2" "
#include <emmintrin.h>
int main() {
    __m128i a = _mm_set_epi32 (-1, 2, -3, 4);
    __m128i result = _mm_abs_epi32 (a);
    return 0;
}" SIMD_SSE2)

# Check SSE3 (not a valid flag for MSVC)
check_simd_capability("-msse3" "" "SSE3" "
#include <pmmintrin.h>
int main() {
    __m128 a = _mm_set_ps (-1.0f, 2.0f, -3.0f, 4.0f);
    __m128 b = _mm_set_ps (1.0f, 2.0f, 3.0f, 4.0f);
    __m128 result = _mm_addsub_ps (a, b);
    return 0;
}" SIMD_SSE3)

# Check SSSE3 (not a valid flag for MSVC)
check_simd_capability("-mssse3" "" "SSSE3" "
#include <tmmintrin.h>
int main() {
    __m128i a = _mm_set_epi8(-1, 2, -3, 4, -1, 2, -3, 4, -1, 2, -3, 4, -1, 2, -3, 4);
    __m128i result = _mm_abs_epi8(a);
    return 0;
}" SIMD_SSSE3)

# Check SSE4.1 (not a valid flag for MSVC)
check_simd_capability("-msse4.1" "" "SSE4.1" "
#include <smmintrin.h>
int main() {
    __m128i a = _mm_set_epi32(-1, 2, -3, 4);
    __m128i result = _mm_abs_epi32(a);
    return 0;
}" SIMD_SSE4_1)

# Check SSE4.2 (not a valid flag for MSVC)
check_simd_capability("-msse4.2" "" "SSE4.2" "
#include <nmmintrin.h>
int main() {
    __m128i a = _mm_set_epi32(-1, 2, -3, 4);
    __m128i result = _mm_abs_epi32(a);
    return 0;
}" SIMD_SSE4_2)

check_simd_capability("-mfma" "/arch:FMA" "FMA3" "
#include <immintrin.h>
int main() {
    __m256 a = _mm256_set_ps(-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 c = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    __m256 result = _mm256_fmadd_ps(a, b, c);
    return 0;
}" SIMD_FMA3)

# Check AVX
check_simd_capability("-mavx" "/arch:AVX" "AVX" "
#include <immintrin.h>
int main() {
    __m256 a = _mm256_set_ps(-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 result = _mm256_abs_ps(a);
    return 0;
}" SIMD_AVX)

# Check AVX2
check_simd_capability("-mavx2" "/arch:AVX2" "AVX2" "
#include <immintrin.h>
int main() {
    __m256i a = _mm256_set_epi32(-1, 2, -3, 4, -1, 2, -3, 4);
    __m256i result = _mm256_abs_epi32(a);
    return 0;
}" SIMD_AVX2)

# Check AVX512F
check_simd_capability("-mavx512f" "/arch:AVX512" "AVX512F" "
#include <immintrin.h>
int main() {
    __m512i a = _mm512_set_epi32(-1, 2, -3, 4, -1, 2, -3, 4, -1, 2, -3, 4, -1, 2, -3, 4);
    __m512i result = _mm512_abs_epi32(a);
    return 0;
}" SIMD_AVX512F)

# Check AVX512BW
check_simd_capability("-mavx512bw" "/arch:AVX512" "AVX512BW" "
#include <immintrin.h>
int main() {
    __m512i a = _mm512_set_epi64(-1, 2, -3, 4, -1, 2, -3, 4);
    __m512i result = _mm512_abs_epi8(a);
    return 0;
}" SIMD_AVX512BW)

# Check AVX512CD
check_simd_capability("-mavx512cd" "/arch:AVX512" "AVX512CD" "
#include <immintrin.h>
int main() {
    __m512i a = _mm512_set_epi64(-1, 2, -3, 4, -1, 2, -3, 4);
    __m512i result = _mm512_conflict_epi64(a);
    return 0;
}" SIMD_AVX512CD)

# Check AVX512DQ
check_simd_capability("-mavx512dq" "/arch:AVX512" "AVX512DQ" "
#include <immintrin.h>
int main() {
    __m512d a = _mm512_set_pd(-1.0, 2.0, -3.0, 4.0, -1.0, 2.0, -3.0, 4.0);
    __m512d result = _mm512_abs_pd(a);
    return 0;
}" SIMD_AVX512DQ)

# Check AVX512ER
check_simd_capability("-mavx512er" "/arch:AVX512" "AVX512ER" "
#include <immintrin.h>
int main() {
    __m512d a = _mm512_set_pd(-1.0, 2.0, -3.0, 4.0, -1.0, 2.0, -3.0, 4.0);
    __m512d result = _mm512_exp_pd(a);
    return 0;
}" SIMD_AVX512ER)

# Check AVX512PF
check_simd_capability("-mavx512pf" "/arch:AVX512" "AVX512PF" "
#include <immintrin.h>
int main() {
    __m512 a = _mm512_set_ps(-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m512 result = _mm512_exp_ps(a);
    return 0;
}" SIMD_AVX512PF)

check_simd_capability("-march=armv6" "" "ARMv6" "
#include <arm_neon.h>
int main() {
    int32x2_t a = vdup_n_s32(1);
    int32x2_t b = vdup_n_s32(2);
    int32x2_t result = vadd_s32(a, b);
    return 0;
}" SIMD_ARMv6)

# ARM
check_simd_capability("-march=armv7-a" "" "ARMv7" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv7)

check_simd_capability("-march=armv8-a" "" "ARMv8" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8)

# ARM64
check_simd_capability("-march=armv8.1-a" "" "ARMv8.1" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8_1)

check_simd_capability("-march=armv8.2-a" "" "ARMv8.2" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8_2)

check_simd_capability("-march=armv8.3-a" "" "ARMv8.3" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8_3)

check_simd_capability("-march=armv8.4-a" "" "ARMv8.4" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8_4)

check_simd_capability("-march=armv8.5-a" "" "ARMv8.5" "
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(1);
    int32x4_t b = vdupq_n_s32(2);
    int32x4_t result = vaddq_s32(a, b);
    return 0;
}" SIMD_ARMv8_5)

if (LIBRAPID_ARCH_FOUND)
    message(STATUS "[ LIBRAPID ] Architecture Flags: ${LIBRAPID_ARCH_FLAGS}")
else ()
    message(STATUS "[ LIBRAPID ] Architecture Flags Not Found")
endif ()
