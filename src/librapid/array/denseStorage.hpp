#pragma once

#include "../internal/config.hpp"
#include "../internal/memUtils.hpp"
#include "../cuda/memUtils.hpp"
#include "valueReference.hpp"
#include "../math/coreMath.hpp"

namespace librapid::memory {
	template<typename T, typename d>
	class DenseStorage {
	public:
		using Type = T;
		friend DenseStorage<bool, d>;

		DenseStorage() = default;

		explicit DenseStorage(int64_t size) :
				m_size(size), m_heap(memory::malloc<T, d>(size)),
				m_refCount(new std::atomic<int64_t>(1)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		DenseStorage(const DenseStorage<T, d> &other) {
			m_refCount = other.m_refCount;
			m_size	   = other.m_size;
			m_heap	   = other.m_heap;
			increment();
		}

		DenseStorage &operator=(const DenseStorage<T, d> &other) {
			if (this == &other) return *this;

			decrement();
			m_size	   = other.m_size;
			m_heap	   = other.m_heap;
			m_refCount = other.m_refCount;
			increment();

			return *this;
		}

		~DenseStorage() { decrement(); }

		operator bool() const { return (bool)m_refCount; }

		ValueReference<T, d> operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heap + index);
		}

		ValueReference<T, d> operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heap + index);
		}

		void setSize_(int64_t newSize) { m_size = newSize; }

		LR_NODISCARD("") T *heap() const { return m_heap; }

		LR_NODISCARD("") int64_t size() const { return m_size; }

		LR_NODISCARD("") int64_t bytes() const { return sizeof(T) * m_size; }

		void increment() const { (*m_refCount)++; }

		void decrement() {
			if (!m_refCount) return;
			(*m_refCount)--;
			if (*m_refCount == 0) {
				delete m_refCount;
				memory::free<T, d>(m_heap);
			}
		}

	protected:
		int64_t m_size					 = 0;
		T *m_heap						 = nullptr;
		std::atomic<int64_t> *m_refCount = nullptr;
	};

	template<typename d>
	class DenseStorage<bool, d> : public DenseStorage<uint64_t, d> {
	public:
		using Type = bool;
		using Base = DenseStorage<uint64_t, d>;

		DenseStorage() : Base() {};

		explicit DenseStorage(int64_t size) : Base((size + 512) / 64) {
			this->m_size = size;
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		ValueReference<bool, d> operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < this->m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  this->m_size);
			uint64_t block = index / 64;
			uint16_t bit   = mod<uint64_t>(index, 64);
			return ValueReference<bool, d>(this->m_heap + block, bit);
		}

		ValueReference<bool, d> operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < this->m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  this->m_size);
			uint64_t block = index / 64;
			uint16_t bit   = mod<uint64_t>(index, 64);
			return ValueReference<bool, d>(this->m_heap + block, bit);
		}
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst, const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between DenseStorage objects with different sizes");
		memcpy<T, d, T_, d_>(dst.heap(), src.heap(), dst.size());
	}
} // namespace librapid::memory
