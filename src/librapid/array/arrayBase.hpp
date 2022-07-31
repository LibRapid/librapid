#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/extent.hpp"
#include "functors/functors.hpp"
#include "cast.hpp"

#define IMPL_BINOP(NAME, TYPE)                                                                     \
	template<typename OtherDerived,                                                                \
			 bool forceTemporary													   = false,    \
			 typename std::enable_if_t<!internal::traits<OtherDerived>::IsScalar, int> = 0>        \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherDerived &other) const {                                                   \
		using ScalarOther = typename internal::traits<OtherDerived>::Scalar;                       \
		using OtherDevice = typename internal::traits<OtherDerived>::Device;                       \
		using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;             \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, ScalarOther>, Derived, OtherDerived>;   \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                      \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
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
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, OtherScalar>, Derived, OtherScalar>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), other).eval();                                               \
		else                                                                                       \
			return RetType(derived(), other);                                                      \
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
		  binop::CWiseBinop<functors::binary::TYPE<OtherScalar, Scalar>, OtherScalar, Derived>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(other, arr.derived()).eval();                                           \
		else                                                                                       \
			return RetType(other, arr.derived());                                                  \
	}

#define IMPL_UNOP(NAME, TYPE)                                                                        \
	LR_NODISCARD("")                                                                                 \
	auto NAME() const {                                                                              \
		using RetType					   = unop::CWiseUnop<functors::unop::TYPE<Scalar>, Derived>; \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                        \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;        \
                                                                                                     \
		static_assert(!(Required & ~(Flags & Required)),                                             \
					  "Scalar type is incompatible with Functor");                                   \
                                                                                                     \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))               \
			return RetType(derived()).eval();                                                        \
		else                                                                                         \
			return RetType(derived());                                                               \
	}

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			static constexpr bool IsScalar	= false;
			using Valid						= std::true_type;
			using Scalar					= typename traits<Derived>::Scalar;
			using BaseScalar				= typename traits<Scalar>::BaseScalar;
			using Device					= device::CPU;
			using StorageType				= memory::DenseStorage<Scalar, device::CPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags | flags::PythonFlags;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			static constexpr bool IsScalar	= false;
			using Valid						= std::true_type;
			using Scalar					= typename traits<Derived>::Scalar;
			using BaseScalar				= typename traits<Scalar>::BaseScalar;
			using Device					= device::CPU;
			using StorageType				= memory::DenseStorage<Scalar, device::GPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags | flags::PythonFlags;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar	  = typename internal::traits<Derived>::Scalar;
		using BaseScalar  = typename internal::traits<Scalar>::BaseScalar;
		using This		  = ArrayBase<Derived, Device>;
		using Packet	  = typename internal::traits<Derived>::Packet;
		using StorageType = typename internal::traits<Derived>::StorageType;
		// using ArrayExtent = ExtentType < int64_t, 32,
		// 	  internal::traits<Scalar>::PacketWidth<4 ? 4 : internal::traits<Scalar>::PacketWidth>;
		using ArrayExtent				= Extent; // ExtentType<int64_t, 32, 1>;
		static constexpr uint64_t Flags = internal::traits<This>::Flags;

		friend Derived;

		ArrayBase() = default;

		template<typename T_, int64_t d_, int64_t a_>
		explicit ArrayBase(const ExtentType<T_, d_, a_> &extent) :
				m_isScalar(extent.size() == 0), m_extent(extent), m_storage(extent.sizeAdjusted()) {
		}

		template<typename T_, int64_t d_, int64_t a_>
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
void castKernel({1} *dst, {2} *src, int64_t size) {{
	const int64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < size) {{
		dst[kernelIndex] = ({1}) src[kernelIndex];
	}}
}}
				)V0G0N",
												 headers,
												 internal::traits<T>::Name,
												 internal::traits<Scalar>::Name);

				int64_t elems = m_extent.sizeAdjusted();

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, nvccOptions);

				int64_t threadsPerBlock, blocksPerGrid;

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
			int64_t size = m_extent.sizeAdjusted();

			if constexpr (std::is_same_v<Scalar, bool>) {
				size += sizeof(BaseScalar) * 8;
				size /= sizeof(BaseScalar) * 8;
			}

			fmt::print("Information: {}\n", typeid(BaseScalar).name());

			memory::memcpy<BaseScalar, D, BaseScalar, Device>(
			  res.storage().heap(), eval().storage().heap(), size);
			return res;
		}

		template<typename T, typename D>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto castMove() const {
			Array<Scalar, D> res(m_extent);
			int64_t size = m_extent.sizeAdjusted();

			if constexpr (std::is_same_v<Scalar, bool>) {
				size += sizeof(BaseScalar) * 8;
				size /= sizeof(BaseScalar) * 8;
			}

			memory::memcpy<BaseScalar, D, BaseScalar, Device>(
			  res.storage().heap(), eval().storage().heap(), size);
			return res.template cast<T>();
		}

		IMPL_BINOP(operator+, ScalarSum)
		IMPL_BINOP(operator-, ScalarDiff)
		IMPL_BINOP(operator*, ScalarProd)
		IMPL_BINOP(operator/, ScalarDiv)

		IMPL_BINOP_SCALAR(operator+, ScalarSum)
		IMPL_BINOP_SCALAR(operator-, ScalarDiff)
		IMPL_BINOP_SCALAR(operator*, ScalarProd)
		IMPL_BINOP_SCALAR(operator/, ScalarDiv)

		IMPL_BINOP(operator|, BitwiseOr)
		IMPL_BINOP(operator&, BitwiseAnd)
		IMPL_BINOP(operator^, BitwiseXor)

		IMPL_UNOP(operator-, UnaryMinus)
		IMPL_UNOP(operator~, BitwiseNot)
		IMPL_UNOP(operator!, UnaryNot)

		auto transposed(const Extent &order_ = {}) const {
			using RetType = unop::CWiseUnop<functors::matrix::Transpose<Derived>, Derived>;
			static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;
			static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;

			static_assert(!(Required & ~(Flags & Required)),
						  "Scalar type is incompatible with Functor");

			ArrayExtent order;
			if (order_.dims() == -1) {
				// Default order is to reverse all indices
				order = ArrayExtent::zero(m_extent.dims());
				for (int64_t i = 0; i < m_extent.dims(); ++i) {
					order[m_extent.dims() - i - 1] = i;
				}
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
				int64_t threadNum = omp_get_thread_num();
#	else
				int64_t threadNum = 0;
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
			using DeviceOther = typename internal::traits<OtherDerived>::Device;

			if constexpr (std::is_same_v<Device, device::CPU> &&
						  std::is_same_v<DeviceOther, device::GPU>)
				return move<DeviceOther>().dot(other);
			else if constexpr (std::is_same_v<Device, device::GPU> &&
							   std::is_same_v<DeviceOther, device::CPU>)
				return dot(other.template move<Device>());

			if (m_extent.dims() == 1 && other.m_extent.dims() == 1) {
				// Vector dot product
				auto strideThis	 = m_extent.strideAdjusted();
				auto strideOther = other.extent().strideAdjusted();

				auto res = blas::dot<Device>(m_extent.sizeAdjusted(),
											 m_storage.heap(),
											 strideThis[0],
											 other.storage().heap(),
											 strideOther[0]);

				return Array<Scalar, Device>(res);
			}

			return Array<Scalar, Device>(0);
		}

		LR_NODISCARD("Do not ignore the result of an evaluated calculation")
		auto eval() const { return derived(); }

		template<typename OtherDerived>
		LR_FORCE_INLINE void loadFrom(int64_t index, const OtherDerived &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writePacket(index, other.packet(index));
		}

		template<typename ScalarType>
		LR_FORCE_INLINE void loadFromScalar(int64_t index, const ScalarType &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writeScalar(index, other.scalar(index));
		}

		LR_FORCE_INLINE Derived &assign(const Scalar &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent   = ArrayExtent(1);
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

			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			m_isScalar = other.isScalar();

			using Selector = functors::AssignSelector<Derived, OtherDerived, false>;
			return Selector::run(derived(), other.derived());
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assignLazy(const OtherDerived &other) {
			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			using Selector = functors::AssignOp<Derived, OtherDerived>;
			Selector::run(derived(), other.derived());
			return derived();
		}

		LR_NODISCARD("") LR_FORCE_INLINE const Derived &derived() const {
			return *static_cast<const Derived *>(this);
		}

		LR_FORCE_INLINE Packet packet(int64_t index) const {
			Packet p;
			if constexpr (is_same_v<Scalar, bool>)
				p.load(m_storage.heap() + (index / 64));
			else
				p.load(m_storage.heap() + index);
			return p;
		}

		LR_FORCE_INLINE Scalar scalar(int64_t index) const { return m_storage[index].get(); }

		template<typename T>
		std::string genKernel(std::vector<T> &vec, int64_t &index) const {
			vec.emplace_back((T)m_storage.heap());
			return fmt::format("arg{}", index++);
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived &derived() {
			return *static_cast<Derived *>(this);
		}

		LR_NODISCARD("") bool isScalar() const { return m_isScalar; }
		LR_NODISCARD("") const StorageType &storage() const { return m_storage; }
		LR_NODISCARD("") StorageType &storage() { return m_storage; }
		LR_NODISCARD("") ArrayExtent extent() const { return m_extent; }
		LR_NODISCARD("") ArrayExtent &extent() { return m_extent; }

	private:
		bool m_isScalar = false;
		ArrayExtent m_extent;
		StorageType m_storage;
	};

	IMPL_BINOP_SCALAR_EXTERNAL(operator+, ScalarSum)
	IMPL_BINOP_SCALAR_EXTERNAL(operator-, ScalarDiff)
	IMPL_BINOP_SCALAR_EXTERNAL(operator*, ScalarProd)
	IMPL_BINOP_SCALAR_EXTERNAL(operator/, ScalarDiv)
} // namespace librapid

#undef IMPL_BINOP
#undef IMPL_BINOP_SCALAR
#undef IMPL_BINOP_SCALAR_EXTERNAL
#undef IMPL_UNOP