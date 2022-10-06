#pragma once

#define IMPL_BINOP(NAME, TYPE)                                                                     \
	template<bool forceTemporary = false,                                                          \
			 typename OtherDerived,                                                                \
			 typename std::enable_if_t<!internal::traits<OtherDerived>::IsScalar, int> = 0>        \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherDerived &other) const {                                                   \
		using ScalarOther = typename internal::traits<OtherDerived>::Scalar;                       \
		using OtherDevice = typename internal::traits<OtherDerived>::Device;                       \
		using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;             \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, ScalarOther>, Derived, OtherDerived>;   \
		static constexpr ui64 Flags	   = internal::traits<Scalar>::Flags;                          \
		static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;          \
                                                                                                   \
		static_assert(                                                                             \
		  is_same_v<Scalar, ScalarOther>,                                                          \
		  "Cannot operate on Arrays with different types. Please use Array::cast<T>()");           \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		LR_ASSERT(extent() == other.extent(),                                                      \
				  "Arrays must have equal extents. Cannot operate on Arrays with {} and {}",       \
				  extent().str(),                                                                  \
				  other.extent().str());                                                           \
                                                                                                   \
		if constexpr (!forceTemporary && /* If we REQUIRE a temporary value, don't evaluate it */  \
					  ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval)))           \
			return RetType(derived(), other.derived()).eval();                                     \
		else                                                                                       \
			return RetType(derived(), other.derived());                                            \
	}

#define IMPL_BINOP_SCALAR(NAME, TYPE)                                                              \
	template<typename OtherScalar,                                                                 \
			 typename std::enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherScalar &other) const {                                                    \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, Scalar>, Derived, Scalar>;              \
		static constexpr ui64 Flags	   = internal::traits<OtherScalar>::Flags;                     \
		static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;          \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), static_cast<Scalar>(other)).eval();                          \
		else                                                                                       \
			return RetType(derived(), static_cast<Scalar>(other));                                 \
	}

#define IMPL_BINOP_SCALAR_EXTERNAL(NAME, TYPE)                                                     \
	template<typename OtherScalar,                                                                 \
			 typename Derived,                                                                     \
			 typename Device,                                                                      \
			 typename std::enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	LR_INLINE auto NAME(const OtherScalar &other, const ArrayBase<Derived, Device> &arr) {         \
		using Scalar	= typename internal::traits<Derived>::Scalar;                              \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, Scalar>, Scalar, Derived>;              \
		static constexpr ui64 Flags	   = internal::traits<Scalar>::Flags;                          \
		static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;          \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(Scalar(other), arr.derived()).eval();                                   \
		else                                                                                       \
			return RetType(Scalar(other), arr.derived());                                          \
	}

#define IMPL_UNOP(NAME, TYPE, OVERRIDE_BOOL)                                                         \
	template<bool forceTemporary = false>                                                            \
	LR_NODISCARD("")                                                                                 \
	auto NAME() const {                                                                              \
		if constexpr (std::is_same_v<Scalar, bool> && OVERRIDE_BOOL) {                               \
			return operator~();                                                                      \
		} else {                                                                                     \
			using RetType				   = unop::CWiseUnop<functors::unop::TYPE<Scalar>, Derived>; \
			static constexpr ui64 Flags	   = internal::traits<Scalar>::Flags;                        \
			static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;        \
                                                                                                     \
			static_assert(!(Required & ~(Flags & Required)),                                         \
						  "Scalar type is incompatible with Functor");                               \
                                                                                                     \
			if constexpr (!forceTemporary && /* If a temporary value is required, don't eval */      \
						  ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval)))         \
				return RetType(derived()).eval();                                                    \
			else                                                                                     \
				return RetType(derived());                                                           \
		}                                                                                            \
	}

#define IMPL_UNOP_EXTERNAL(NAME, TYPE)                                                             \
	template<bool forceTemporary = false, typename Derived, typename Device>                       \
	LR_NODISCARD("")                                                                               \
	auto NAME(const ArrayBase<Derived, Device> &arr) {                                             \
		using Scalar				   = typename internal::traits<Derived>::Scalar;               \
		using RetType				   = unop::CWiseUnop<functors::unop::TYPE<Scalar>, Derived>;   \
		static constexpr ui64 Flags	   = internal::traits<Scalar>::Flags;                          \
		static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;          \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr (!forceTemporary && /* If a temporary value is required, don't eval */        \
					  ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval)))           \
			return RetType(arr.derived()).eval();                                                  \
		else                                                                                       \
			return RetType(arr.derived());                                                         \
	}

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = traits<Derived>::IsEvaluated;
			using Valid						  = std::true_type;
			using Scalar					  = typename traits<Derived>::Scalar;
			using BaseScalar				  = typename traits<Scalar>::BaseScalar;
			using Device					  = device::CPU;
			using StorageType				  = memory::DenseStorage<Scalar, device::CPU>;
			static constexpr ui64 Flags		  = traits<Derived>::Flags | flags::PythonFlags;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = traits<Derived>::IsEvaluated;
			using Valid						  = std::true_type;
			using Scalar					  = typename traits<Derived>::Scalar;
			using BaseScalar				  = typename traits<Scalar>::BaseScalar;
			using Device					  = device::CPU;
			using StorageType				  = memory::DenseStorage<Scalar, device::GPU>;
			static constexpr ui64 Flags		  = traits<Derived>::Flags | flags::PythonFlags;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar				= typename internal::traits<Derived>::Scalar;
		using BaseScalar			= typename internal::traits<Scalar>::BaseScalar;
		using This					= ArrayBase<Derived, Device>;
		using Packet				= typename internal::traits<Derived>::Packet;
		using StorageType			= typename internal::traits<Derived>::StorageType;
		static constexpr ui64 Flags = internal::traits<This>::Flags;

		friend Derived;

		ArrayBase() = default;

		template<typename T_, i32 d_, i32 a_>
		explicit ArrayBase(const ExtentType<T_, d_, a_> &extent) :
				m_isScalar(extent.size() == 0), m_extent(extent), m_storage(extent.sizeAdjusted()) {
		}

		template<typename T_, i32 d_, i32 a_>
		explicit ArrayBase(const ExtentType<T_, d_, a_> &extent, int) :
				m_isScalar(extent.size() == 0), m_extent(extent) {}

		ArrayBase(const Array<Scalar, Device> &other) {
			m_isScalar = other.isScalar();
			m_extent   = other.extent();
			m_storage  = other.storage();
		}

		template<typename OtherDerived>
		ArrayBase(const OtherDerived &other) {
			assign(other);
		}

		template<typename OtherDerived>
		LR_INLINE Derived &operator=(const OtherDerived &other) {
			return assign(other);
		}

		template<typename T>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto cast() const {
			using ScalarType = typename internal::traits<T>::Scalar;
			using RetType	 = unary::Cast<ScalarType, Derived>;

			if constexpr (std::is_same_v<Device, device::CPU>) {
				if constexpr ((bool)(internal::traits<RetType>::Flags &
									 internal::flags::RequireEval))
					return RetType(derived()).eval();
				else
					return RetType(derived());
			}
#if defined(LIBRAPID_HAS_CUDA)
			else {
				std::string headers;
				for (const auto &header : customHeaders) {
					headers += fmt::format("#include \"{}\"\n", header);
				}

				std::string kernel = fmt::format(R"V0G0N(castingKernel
#include <stdint.h>

{0}

__global__
void castKernel({1} *dst, {2} *src, i64 size) {{
	const i64 kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < size) {{
		dst[kernelIndex] = ({1}) src[kernelIndex];
	}}
}}
				)V0G0N",
												 headers,
												 internal::traits<T>::Name,
												 internal::traits<Scalar>::Name);

				i64 elems = m_extent.sizeAdjusted();

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, nvccOptions);

				i64 threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = elems;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				Array<T, Device> res(m_extent);

				jitifyCall(program.kernel("castKernel")
							 .instantiate()
							 .configure(grid, block, 0, memory::cudaStream)
							 .launch(res.storage().heap(), m_storage.heap(), elems));
				return res;
			}
#else
			else {
				LR_ASSERT(false, "CUDA support not enabled");
			}
#endif
		}

		template<typename D>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto move() const {
			Array<Scalar, D> res(m_extent);
			i64 size = m_extent.sizeAdjusted();

			// if constexpr (std::is_same_v<Scalar, bool>) {
			// 	size += sizeof(BaseScalar) * 8;
			// 	size /= sizeof(BaseScalar) * 8;
			// }

			memory::memcpy<BaseScalar, D, BaseScalar, Device>(
			  res.storage().heap(), eval().storage().heap(), size);
			return res;
		}

		template<typename T, typename D>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto castMove() const {
			Array<Scalar, D> res(m_extent);
			i64 size = m_extent.sizeAdjusted();

			// if constexpr (std::is_same_v<Scalar, bool>) {
			// 	size += sizeof(BaseScalar) * 8;
			// 	size /= sizeof(BaseScalar) * 8;
			// }

			memory::memcpy<BaseScalar, D, BaseScalar, Device>(
			  res.storage().heap(), eval().storage().heap(), size);
			return res.template cast<T>();
		}

		IMPL_BINOP(operator+, ScalarSum)
		IMPL_BINOP(operator-, ScalarDiff)
		IMPL_BINOP(operator*, ScalarProd)
		IMPL_BINOP(operator/, ScalarDiv)

		// Allow bool as operand
		IMPL_BINOP_SCALAR(operator+, ScalarSum)
		IMPL_BINOP_SCALAR(operator-, ScalarDiff)
		IMPL_BINOP_SCALAR(operator*, ScalarProd)
		IMPL_BINOP_SCALAR(operator/, ScalarDiv)

		IMPL_BINOP(operator|, BitwiseOr)
		IMPL_BINOP(operator&, BitwiseAnd)
		IMPL_BINOP(operator^, BitwiseXor)

		// Do not allow bool as operator (Can be done with other operations)
		IMPL_BINOP_SCALAR(operator|, BitwiseOr)
		IMPL_BINOP_SCALAR(operator&, BitwiseAnd)
		IMPL_BINOP_SCALAR(operator^, BitwiseXor)

		IMPL_UNOP(operator-, UnaryMinus, true)
		IMPL_UNOP(operator~, UnaryInvert, false)
		IMPL_UNOP(operator!, UnaryNot, true)

		auto transposed(const Extent &order_ = {}) const {
			using RetType = unop::CWiseUnop<functors::matrix::Transpose<Derived>, Derived>;
			static constexpr ui64 Flags	   = internal::traits<Scalar>::Flags;
			static constexpr ui64 Required = RetType::Flags & internal::flags::OperationMask;

			static_assert(!(Required & ~(Flags & Required)),
						  "Scalar type is incompatible with Functor");

			Extent order;
			if (order_.dims() == -1) {
				// Default order is to reverse all indices
				order = Extent::zero(m_extent.dims());
				for (i64 i = 0; i < m_extent.dims(); ++i) { order[m_extent.dims() - i - 1] = i; }
			} else {
				order = order_;
			}

			if constexpr (std::is_same_v<Device, device::CPU>) {
				return RetType(derived(), order);
			}
#if defined(LIBRAPID_HAS_CUDA)
			else {
				// Access the handle for the corresponding thread. As CUBLAS handles are not
				// thread-safe (when in use) we need a separate handle per thread.
#	if defined(LIBRAPID_HAS_OMP)
				i64 threadNum = omp_get_thread_num();
#	else
				i64 threadNum = 0;
#	endif

				Array<Scalar, Device> res(m_extent.swivelled(order));

				if constexpr (std::is_same_v<Scalar, extended::float16_t>) {
					auto tmp = cast<float>();
					return tmp.transposed(order).template cast<extended::float16_t>();
				} else if constexpr (std::is_same_v<Scalar, float>) {
					float alpha = 1;
					float beta	= 0;
					cublasSafeCall(cublasSgeam(memory::cublasHandles[threadNum],
											   CUBLAS_OP_T,
											   CUBLAS_OP_N,
											   m_extent.adjusted(0),
											   m_extent.adjusted(1),
											   &alpha,
											   m_storage.heap(),
											   m_extent.adjusted(1),
											   &beta,
											   nullptr,
											   m_extent.adjusted(1),
											   res.storage().heap(),
											   m_extent.adjusted(0)));
				}

				return res;
			}
#endif
		}

		template<typename OtherDerived>
		auto dot(const OtherDerived &other) const {
			// Should this always return an evaluated result???
			using ScalarThis  = typename internal::traits<Derived>::BaseScalar;
			using ScalarOther = typename internal::traits<OtherDerived>::BaseScalar;
			using DeviceOther = typename internal::traits<OtherDerived>::Device;

			// Ensure Boolean arrays cannot be used with dot product
			static_assert(!std::is_same_v<typename internal::traits<Derived>::Scalar, bool> &&
							!std::is_same_v<typename internal::traits<OtherDerived>::Scalar, bool>,
						  "Dot product not defined for boolean arrays");

			if constexpr (std::is_same_v<Device, device::CPU> &&
						  std::is_same_v<DeviceOther, device::GPU>)
				return move<DeviceOther>().dot(other);
			else if constexpr (std::is_same_v<Device, device::GPU> &&
							   std::is_same_v<DeviceOther, device::CPU>)
				return dot(other.template move<Device>());

			if constexpr (!internal::traits<Derived>::IsEvaluated ||
						  !internal::traits<OtherDerived>::IsEvaluated) {
				return eval().dot(other.eval());
			}

			if (m_extent.dims() == 1 && other.extent().dims() == 1) {
				// Vector dot product
				auto strideThis	 = m_extent.strideAdjusted();
				auto strideOther = other.extent().strideAdjusted();

				auto res = blas::dot<Device>(m_extent.sizeAdjusted(),
											 m_storage.heap(),
											 strideThis[0],
											 other.storage().heap(),
											 strideOther[0]);

				return Array<Scalar, Device>(res);
			} else if (m_extent.dims() == 2 && other.extent().dims() == 1) {
				// Matrix-vector product

				LR_ASSERT(m_extent[1] == other.extent()[0],
						  "Columns of left matrix must match elements of right matrix");

				i64 m = m_extent[0]; // Rows of left matrix
				i64 n = m_extent[1]; // Columns of left matrix

				auto res = Array<Scalar, Device>(Extent(m));

				blas::gemv<Device>(false,
								   m,
								   n,
								   Scalar(1),
								   m_storage.heap(),
								   m_extent.strideAdjusted()[0],
								   other.storage().heap(),
								   other.extent().strideAdjusted()[0],
								   Scalar(0),
								   res.storage().heap(),
								   res.extent().strideAdjusted()[0]);

				return res;
			} else if (m_extent.dims() == 2 && other.extent().dims() == 2) {
				// Matrix product

				LR_ASSERT(m_extent[1] == other.extent()[0],
						  "Columns of left matrix must match rows of right matrix");

				i64 m = m_extent[0];	   // Rows of left matrix
				i64 n = other.extent()[1]; // Columns of right matrix
				i64 k = m_extent[1];	   // Columns of left matrix

				Array<Scalar, Device> res(Extent(m, n));
				blas::gemm<Device>(false,
								   false,
								   m,
								   n,
								   k,
								   ScalarThis(1),
								   m_storage.heap(),
								   m_extent.strideAdjusted()[0],
								   other.storage().heap(),
								   other.extent().strideAdjusted()[0],
								   ScalarOther(0),
								   res.storage().heap(),
								   res.extent().strideAdjusted()[0]);

				return res;
			}

			return Array<Scalar, Device>(Scalar(0));
		}

		LR_NODISCARD("Do not ignore the result of an evaluated calculation")
		auto eval() const {
			if (Flags & internal::flags::Evaluated)
				return *this;
			return derived().eval();
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE void loadFrom(i64 index, const OtherDerived &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writePacket(index, other.packet(index));
		}

		template<typename ScalarType>
		LR_FORCE_INLINE void loadFromScalar(i64 index, const ScalarType &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writeScalar(index, other.scalar(index));
		}

		LR_FORCE_INLINE Derived &assign(const Scalar &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent   = Extent(1);
				m_storage  = StorageType(m_extent.sizeAdjusted());
				m_isScalar = true;
			}

			LR_ASSERT(m_isScalar, "Cannot assign Scalar to non-scalar Array");
			m_storage[0] = other;
			return derived();
		}

		LR_FORCE_INLINE Derived &assign(const Array<Scalar, Device> &other) {
			m_extent   = other.extent();
			m_isScalar = other.isScalar();
			m_storage  = other.storage();
			return derived();
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assign(const OtherDerived &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent  = other.extent();
				m_storage = StorageType(m_extent.sizeAdjusted());
			}

			constexpr ui64 flags = internal::traits<OtherDerived>::Flags;
			LR_ASSERT((flags & internal::flags::MatrixTranspose) || (m_extent == other.extent()),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			m_isScalar = other.isScalar();

			using Selector = functors::AssignSelector<Derived, OtherDerived, false>;
			return Selector::run(derived(), other.derived());
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assignLazy(const OtherDerived &other) {
			constexpr ui64 flags = internal::traits<OtherDerived>::Flags;
			LR_ASSERT((flags & internal::flags::MatrixTranspose) || (m_extent == other.extent()),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			// If device differs, we need to copy the data
			if constexpr (!std::is_same_v<Device,
										  typename internal::traits<OtherDerived>::Device>) {
				return assignLazy(other.template move<Device>());
			}

			using Selector = functors::AssignOp<Derived, OtherDerived>;
			Selector::run(derived(), other.derived());
			return derived();
		}

		LR_NODISCARD("") LR_FORCE_INLINE const Derived &derived() const {
			return *static_cast<const Derived *>(this);
		}

		LR_FORCE_INLINE Packet packet(i64 index) const {
			Packet p;
			p.load(m_storage.heap() + index);
			return p;
		}

		LR_FORCE_INLINE Scalar scalar(i64 index) const { return m_storage[index].get(); }

		template<typename T>
		std::string genKernel(std::vector<T> &vec, i64 &index) const {
			vec.emplace_back((T)m_storage.heap());
			return fmt::format("arg{}", index++);
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived &derived() {
			return *static_cast<Derived *>(this);
		}

		LR_NODISCARD("") bool isScalar() const { return m_isScalar; }
		LR_NODISCARD("") const StorageType &storage() const { return m_storage; }
		LR_NODISCARD("") StorageType &storage() { return m_storage; }
		LR_NODISCARD("") Extent extent() const { return m_extent; }
		LR_NODISCARD("") Extent &extent() { return m_extent; }

	private:
		bool m_isScalar = false;
		Extent m_extent;
		StorageType m_storage;
	};

	IMPL_BINOP_SCALAR_EXTERNAL(operator+, ScalarSum)
	IMPL_BINOP_SCALAR_EXTERNAL(operator-, ScalarDiff)
	IMPL_BINOP_SCALAR_EXTERNAL(operator*, ScalarProd)
	IMPL_BINOP_SCALAR_EXTERNAL(operator/, ScalarDiv)

	IMPL_BINOP_SCALAR_EXTERNAL(operator|, BitwiseOr)
	IMPL_BINOP_SCALAR_EXTERNAL(operator&, BitwiseAnd)
	IMPL_BINOP_SCALAR_EXTERNAL(operator^, BitwiseXor)

	IMPL_UNOP_EXTERNAL(sin, Sin)
	IMPL_UNOP_EXTERNAL(cos, Cos)
	IMPL_UNOP_EXTERNAL(tan, Tan)
	IMPL_UNOP_EXTERNAL(asin, Asin)
	IMPL_UNOP_EXTERNAL(acos, Acos)
	IMPL_UNOP_EXTERNAL(atan, Atan)
	IMPL_UNOP_EXTERNAL(sinh, Sinh)
	IMPL_UNOP_EXTERNAL(cosh, Cosh)
	IMPL_UNOP_EXTERNAL(tanh, Tanh)
	IMPL_UNOP_EXTERNAL(asinh, Asinh)
	IMPL_UNOP_EXTERNAL(acosh, Acosh)
	IMPL_UNOP_EXTERNAL(atanh, Atanh)
	IMPL_UNOP_EXTERNAL(exp, Exp)
	IMPL_UNOP_EXTERNAL(log, Log)
	IMPL_UNOP_EXTERNAL(log2, Log2)
	IMPL_UNOP_EXTERNAL(log10, Log10)
	IMPL_UNOP_EXTERNAL(sqrt, Sqrt)
	IMPL_UNOP_EXTERNAL(abs, Abs)
	IMPL_UNOP_EXTERNAL(ceil, Ceil)
	IMPL_UNOP_EXTERNAL(floor, Floor)

} // namespace librapid

#undef IMPL_BINOP
#undef IMPL_BINOP_SCALAR
#undef IMPL_BINOP_SCALAR_EXTERNAL
#undef IMPL_UNOP