#ifndef LIBRAPID_ARRAY
#define LIBRAPID_ARRAY

#include "sizetype.hpp"
#include "strideTools.hpp"
#include "storage.hpp"

#if defined(LIBRAPID_HAS_OPENCL)
#	include "../OpenCL/openclStorage.hpp"
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
#	include "../cuda/cudaStorage.hpp"
#endif // LIBRAPID_HAS_CUDA

#include "arrayTypeDef.hpp"
#include "commaInitializer.hpp"
#include "arrayContainer.hpp"
#include "operations.hpp"
#include "function.hpp"
#include "assignOps.hpp"
#include "arrayView.hpp"
#include "arrayViewString.hpp"
#include "arrayFromData.hpp"
#include "pseudoConstructors.hpp"
#include "fourierTransform.hpp"

#include "linalg/linalg.hpp"

#endif // LIBRAPID_ARRAY