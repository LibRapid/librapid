#pragma once

#include "../internal/config.hpp"
#include "../internal/memUtils.hpp"
#include "../cuda/memUtils.hpp"

namespace librapid::memory {
	template<typename T, typename d>
	class DenseStorage {
		using Type = T;

	public:
		/**
		 * \rst
		 *
		 * Construct an empty ``DenseStorage`` object
		 *
		 * \endrst
		 */
		DenseStorage() = default;

		/**
		 * \rst
		 *
		 * Construct a ``DenseStorage`` object with a given number of elements.
		 *
		 * Parameters
		 * ----------
		 * size: Integer
		 * 		Number of elements to allocate memory for
		 *
		 * \endrst
		 */
		explicit DenseStorage(int64_t size) :
				m_size(size), m_heap(memory::malloc<T, d>(size)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>)
				initializeCudaStream();
#endif
		}

		/**
		 * \rst
		 *
		 * Construct a ``DenseStorage`` object from another instance. The data
		 * is copied exactly.
		 *
		 * Parameters
		 * ----------
		 * other: ``DenseStorage``
		 * 		The instance to copy data from
		 *
		 * \endrst
		 */
		DenseStorage(const DenseStorage<T, d> &other) {
			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			memory::memcpy<T, d>(m_heap, other.m_heap, m_size);
		}

		template<typename T_, typename d_>
		explicit DenseStorage(const DenseStorage<T_, d_> &other) {
			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			memory::memcpy<T, T_, d, d_>(m_heap, other.m_heap, m_size);
		}

		/**
		 * \rst
		 *
		 * Destroy a ``DenseStorage`` object. This will decrement the reference
		 * counter and free the corresponding memory if required.
		 *
		 * \endrst
		 */
		~DenseStorage() { memory::free<T, d>(m_heap); }

		LR_NODISCARD("") T *heap() const { return m_heap; }

		LR_NODISCARD("") int64_t size() const { return m_size; }

		LR_NODISCARD("") int64_t bytes() const { return sizeof(T) * m_size; }

	private:
		int64_t m_size = 0;
		T *m_heap	   = nullptr;
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst,
						  const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between "
				  "DenseStorage objects with "
				  "different sizes");
		memcpy<T, d, T_, d_>(dst.heap(), src.heap(), dst.size());
	}
} // namespace librapid::memory
