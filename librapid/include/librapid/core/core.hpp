#ifndef LIBRAPID_CORE
#define LIBRAPID_CORE

#include "warningSuppress.hpp"
#include "librapidPch.hpp"
#include "debugTrap.hpp"
#include "config.hpp"
#include "global.hpp"
#include "traits.hpp"
#include "typetraits.hpp"
#include "helperMacros.hpp"

#include "forward.hpp"

// BLAS
#include "../cxxblas/cxxblas.h"
#include "../cxxblas/cxxblas.tcc"

// Fourier Transform
#if defined(LIBRAPID_HAS_FFTW) && !defined(LIBRAPID_HAS_CUDA)
// If CUDA is enabled, we use cuFFT
// #	include "../fftw/api/fftw3.h"
#include <fftw3.h>
#endif // LIBRAPID_HAS_CUDA

#pragma warning(push)
#pragma warning(disable: 4324)
#pragma warning(disable: 4458)
#pragma warning(disable: 4456)

#include <pocketfft_hdronly.h>

#pragma warning(pop)

#endif // LIBRAPID_CORE