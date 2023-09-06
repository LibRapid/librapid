#ifndef LIBRAPID_ML_ACTIVATIONS
#define LIBRAPID_ML_ACTIVATIONS

namespace librapid::ml {
    // 1.	[X] Sigmoid . . . . . . f(x) = 1 / (1 + e^-x)
    //								f'(x) = x(1 - x)
    // 2.	[ ] Tanh . . . . . . .  f(x) = tanh(x)
    //								f'(x) = 1 - x^2
    // 3.	[ ] ReLU . . . . . . .  f(x) = max(0, x)
    //								f'(x) = 1 if x > 0 else 0
    // 4.	[ ] LeakyReLU . . . . . f(x) = max(0.01x, x)
    //								f'(x) = 1 if x > 0 else 0.01
    // 5.	[ ] Softmax . . . . . . https://github.com/tiny-dnn/
    //								tiny-dnn/blob/master/tiny_dnn/activations/softmax_layer.h
    // 6.	[ ] Softplus . . . . .  f(x) = ln(1 + e^x)
    //								f'(x) = 1 / (1 + e^-x)
    // 7.	[ ] ELU . . . . . . . . f(x) = x if x > 0 else a(e^x - 1)
    //								f'(x) = 1 if x > 0 else a(e^x)
    // 8.	[ ] SELU . . . . . . .  f(x) = lambda * a * (e^x - 1) if x <= 0 else lambda * x
    //								f'(x) =  lambda * a * e^x if x <= 0 else lambda
    //								α ≈ 1.67326 and λ ≈ 1.0507
    // 9.	[ ] Swish . . . . . . . f(x) = x / (1 + e^-x)
    //								f'(x) = x(1 + e^-x + xe^-x) / (1 + e^-x)^2
    // 10.	[ ] Mish . . . . . . .  f(x) = x * tanh(ln(1 + e^x))
    //								f'(x) = (e^x * (4 * x + 4 + 4 * e^x + e^(2 * x))) / (2 * e^x +
    // e^(2
    //* x)
    //+ 2)^2
    // 11.	[ ] HardSigmoid . . . . f(x) = max(0, min(1, x * 0.2 + 0.5))
    //								f'(x) = 0.2 if 0 < x < 1 else 0
    // 12.	[ ] LogSigmoid . . . .  f(x) = ln(1 / (1 + e^-x))
    //								f'(x) = 1 / (1 + e^x)
    // 13.	[ ] Softsign . . . . .  f(x) = x / (1 + |x|)
    //								f'(x) = 1 / (1 + |x|)^2
    // 14.	[ ] Exponential . . . . f(x) = e^x
    //								f'(x) = e^x
    // 15.	[ ] GELU . . . . . . .  f(x) = x * (1 + erf(x / sqrt(2))) / 2
    //								f'(x) = (erf(x / sqrt(2)) + x * e^(-x^2 / 2) / sqrt(2 * pi)) / 2
    // 18.	[ ] Softmin . . . . . . f(x) = e^x / sum(e^x)
    //								f'(x) = f(x)(1 - f(x))

    /// \brief Sigmoid activation function
    ///
    /// A class that implements the Sigmoid activation function.
    ///
    /// \f$\sigma(x) = \frac{1}{1 + e^{-x}}\f$
    ///
    /// \f$\sigma'(x) = x(1 - x)\f$
    ///
    class Sigmoid {
    public:
        Sigmoid() = default;

        /// Applies the Sigmoid activation function to the input array and returns the result.
        ///
        /// @tparam ShapeType The type of the shape of the input array.
        /// @tparam StorageType The type of the storage of the input array.
        /// @param src The input array to apply the activation function to.
        /// @return A new array with the result of applying the Sigmoid activation function to the
        /// input array.
        template<typename T>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto forward(const T &src) const {
            auto ret = emptyLike(src);
            forward(ret, src);
            return ret;
        }

        template<typename T>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto backward(const T &src) const {
            auto ret = emptyLike(src);
            backward(ret, src);
            return ret;
        }

        template<typename T>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const T &src) const {
            return forward(src);
        }

        /// Applies the Sigmoid activation function to the input array and stores the result in the
        /// output array.
        ///
        /// @tparam ShapeType The type of the shape of the input and output arrays.
        /// @tparam StorageScalar The type of the scalar values stored in the input and output
        /// arrays.
        /// @param dst The output array to store the result of applying the Sigmoid activation
        /// function to the input array.
        /// @param src The input array to apply the activation function to.
        template<typename ShapeType, typename StorageScalar>
        LIBRAPID_ALWAYS_INLINE void
        forward(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst,
                const array::ArrayContainer<ShapeType, Storage<StorageScalar>> &src) const {
            dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
        }

        template<typename ShapeType, typename StorageScalar>
        LIBRAPID_ALWAYS_INLINE void
        backward(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst,
                 const array::ArrayContainer<ShapeType, Storage<StorageScalar>> &src) const {
            dst = src * (StorageScalar(1) - src);
        }

        template<typename ShapeType, typename StorageScalar, size_t... Dims>
        LIBRAPID_ALWAYS_INLINE void forward(
          array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &dst,
          const array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &src) const {
            dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
        }

        template<typename ShapeType, typename StorageScalar, size_t... Dims>
        LIBRAPID_ALWAYS_INLINE void backward(
          array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &dst,
          const array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &src) const {
            dst = src * (StorageScalar(1) - src);
        }

        template<typename ShapeType, typename StorageScalar, typename descriptor, typename Functor,
                 typename... Args>
        LIBRAPID_ALWAYS_INLINE void
        forward(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst,
                const detail::Function<descriptor, Functor, Args...> &src) const {
            dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
        }

        template<typename ShapeType, typename StorageScalar, typename descriptor, typename Functor,
                 typename... Args>
        LIBRAPID_ALWAYS_INLINE void
        backward(array::ArrayContainer<ShapeType, Storage<StorageScalar>> &dst,
                 const detail::Function<descriptor, Functor, Args...> &src) const {
            dst = src * (StorageScalar(1) - src);
        }

#if defined(LIBRAPID_HAS_OPENCL)
        template<
          typename ShapeType, typename StorageScalar, typename Src,
          typename std::enable_if_t<
            std::is_same_v<typename typetraits::TypeInfo<Src>::Backend, backend::OpenCL>, int> = 0>
        LIBRAPID_ALWAYS_INLINE void
        forward(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
                const Src &src) const {
            auto temp = evaluated(src);
            opencl::runLinearKernel<StorageScalar>("sigmoidActivationForward",
                                                   src.shape().size(),
                                                   dst.storage().data(),
                                                   src.storage().data());
        }

        template<
          typename ShapeType, typename StorageScalar, typename Src,
          typename std::enable_if_t<
            std::is_same_v<typename typetraits::TypeInfo<Src>::Backend, backend::OpenCL>, int> = 0>
        LIBRAPID_ALWAYS_INLINE void
        backward(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
                 const Src &src) const {
            auto temp = evaluated(src);
            opencl::runLinearKernel<StorageScalar>("sigmoidActivationBackward",
                                                   temp.shape().size(),
                                                   dst.storage().data(),
                                                   temp.storage().data());
        }
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
        template<
          typename ShapeType, typename StorageScalar, typename Src,
          typename std::enable_if_t<
            std::is_same_v<typename typetraits::TypeInfo<Src>::Backend, backend::CUDA>, int> = 0>
        LIBRAPID_ALWAYS_INLINE void
        forward(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
                const Src &src) const {
            auto temp = evaluated(src);
            cuda::runKernel<StorageScalar, StorageScalar>("activations",
                                                          "sigmoidActivationForward",
                                                          dst.shape().size(),
                                                          temp.shape().size(),
                                                          dst.storage().begin(),
                                                          temp.storage().begin());
        }

        template<
          typename ShapeType, typename StorageScalar, typename Src,
          typename std::enable_if_t<
            std::is_same_v<typename typetraits::TypeInfo<Src>::Backend, backend::CUDA>, int> = 0>
        LIBRAPID_ALWAYS_INLINE void
        backward(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
                 const Src &src) const {
            auto temp = evaluated(src);
            cuda::runKernel<StorageScalar, StorageScalar>("activations",
                                                          "sigmoidActivationBackward",
                                                          dst.shape().size(),
                                                          temp.shape().size(),
                                                          dst.storage().begin(),
                                                          temp.storage().begin());
        }
#endif // LIBRAPID_HAS_CUDA
    };
} // namespace librapid::ml

#endif // LIBRAPID_ML_ACTIVATIONS