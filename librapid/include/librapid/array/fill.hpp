#ifndef LIBRAPID_ARRAY_FILL_HPP
#define LIBRAPID_ARRAY_FILL_HPP

namespace librapid {
    template<typename ShapeType, typename StorageType, typename Scalar>
    LIBRAPID_ALWAYS_INLINE void fill(array::ArrayContainer<ShapeType, StorageType> &dst,
                                     const Scalar &value) {
        dst = array::ArrayContainer<ShapeType, StorageType>(dst.shape(), value);
    }

    template<typename ShapeType, typename StorageScalar, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandom(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst, const Lower &lower,
               const Upper &upper) {
        ShapeType shape = dst.shape();
        auto *data      = dst.storage().begin();
        bool parallel   = global::numThreads != 1 && shape.size() > global::multithreadThreshold;

        if (parallel) {
#pragma omp parallel for
            for (int64_t i = 0; i < shape.size(); ++i) {
                data[i] = random<StorageScalar>(static_cast<StorageScalar>(lower),
                                                static_cast<StorageScalar>(upper));
            }
        } else {
            for (int64_t i = 0; i < shape.size(); ++i) {
                data[i] = random<StorageScalar>(static_cast<StorageScalar>(lower),
                                                static_cast<StorageScalar>(upper));
            }
        }
    }

    template<typename ShapeType, typename StorageScalar, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandomGaussian(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst,
                       const Lower &lower, const Upper &upper) {
        ShapeType shape = dst.shape();
        auto *data      = dst.storage().begin();
        bool parallel   = global::numThreads != 1 && shape.size() > global::multithreadThreshold;

        if (parallel) {
#pragma omp parallel for
            for (int64_t i = 0; i < shape.size(); ++i) {
                data[i] = randomGaussian<StorageScalar>();
            }
        } else {
            for (int64_t i = 0; i < shape.size(); ++i) {
                data[i] = randomGaussian<StorageScalar>();
            }
        }
    }

#if defined(LIBRAPID_HAS_OPENCL)

    template<typename ShapeType, typename StorageScalar, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandom(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
               const Lower &lower, const Upper &upper) {
        ShapeType shape  = dst.shape();
        int64_t elements = shape.size();

        // Initialize a buffer of random seeds
        static int64_t numSeeds = 1024;
        static bool initialized = false;
        static Array<int64_t, backend::OpenCL> seeds(Shape {numSeeds});
        if (global::reseed || !initialized) {
            for (int64_t i = 0; i < numSeeds; ++i) { seeds(i) = randint(0, INT64_MAX); }
            initialized = true;

            // reseed is controlled by the random module, so we don't need to worry about it here
        }

        // Run the kernel
        opencl::runLinearKernel<StorageScalar, true>("fillRandom",
                                                     elements,
                                                     dst.storage().data(),
                                                     elements,
                                                     static_cast<StorageScalar>(lower),
                                                     static_cast<StorageScalar>(upper),
                                                     seeds.storage().data(),
                                                     numSeeds);
    }

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

    template<typename ShapeType, typename StorageScalar, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandom(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
               const Lower &lower, const Upper &upper) {
        ShapeType shape  = dst.shape();
        int64_t elements = shape.size();

        // Initialize a buffer of random seeds
        static int64_t numSeeds = 1024;
        static bool initialized = false;
        static Array<int64_t, backend::CUDA> seeds(Shape {numSeeds});

        if (global::reseed || !initialized) {
            for (int64_t i = 0; i < numSeeds; ++i) { seeds(i) = randint(0, INT64_MAX); }
            initialized = true;

            // reseed is controlled by the random module, so we don't need to worry about it here
        }

        cuda::runKernel<StorageScalar, StorageScalar, StorageScalar>(
          "fill",
          std::is_same_v<StorageScalar, half> ? "fillRandomHalf" : "fillRandom",
          elements,
          dst.storage().data().get(),
          elements,
          static_cast<StorageScalar>(lower),
          static_cast<StorageScalar>(upper),
          seeds.storage().data().get(),
          numSeeds);
    }

    template<typename ShapeType, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandom(array::ArrayContainer<ShapeType, CudaStorage<float>> &dst, const Lower &lower,
               const Upper &upper) {
        ShapeType shape  = dst.shape();
        int64_t elements = shape.size();

        // Create a pseudo-random number generator
        static curandGenerator_t prng;
        static bool initialized = false;

        if (!initialized) {
            curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(prng, global::randomSeed);
            initialized = true;
        }

        if (global::reseed) { curandSetPseudoRandomGeneratorSeed(prng, global::randomSeed); }

        // Run the kernel
        curandGenerateUniform(prng, dst.storage().data().get(), elements);

        // Scale the result to the desired range
        dst = dst * (upper - lower) + lower;
    }

    template<typename ShapeType, typename Lower, typename Upper>
    LIBRAPID_ALWAYS_INLINE void
    fillRandom(array::ArrayContainer<ShapeType, CudaStorage<double>> &dst, const Lower &lower,
               const Upper &upper) {
        ShapeType shape  = dst.shape();
        int64_t elements = shape.size();

        // Create a pseudo-random number generator
        static curandGenerator_t prng;
        static bool initialized = false;

        if (!initialized) {
            curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(prng, global::randomSeed);
            initialized = true;
        }

        if (global::reseed) { curandSetPseudoRandomGeneratorSeed(prng, global::randomSeed); }

        // Run the kernel
        curandGenerateUniformDouble(prng, dst.storage().data().get(), elements);

        // Scale the result to the desired range
        dst = dst * (upper - lower) + lower;
    }

#endif // LIBRAPID_HAS_CUDA
} // namespace librapid

#endif // !LIBRAPID_ARRAY_FILL_HPP