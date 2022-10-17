#pragma once

namespace librapid::memory {
	template<typename T, typename d = device::CPU>
	class DenseStorage {
	public:
		using Type = T;
		friend DenseStorage<bool, d>;

		DenseStorage() = default;

		explicit DenseStorage(size_t size) :
				m_size(roundUpTo(size, internal::traits<T>::PacketWidth)),
				m_heap(memory::malloc<T, d>(m_size)), m_refCount(new std::atomic<i64>(1)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (is_same_v<d, device::GPU>) initializeCudaStream();
#endif
			// This will ALWAYS be set after setting m_heap
			m_heapOffset = m_heap;
		}

		DenseStorage(const DenseStorage &other) :
				m_refCount(other.m_refCount), m_size(other.m_size), m_heap(other.m_heap),
				m_heapOffset(other.m_heapOffset) {
			increment();
		}

		DenseStorage(DenseStorage &&other) noexcept :
				m_refCount(other.m_refCount), m_size(other.m_size), m_heap(other.m_heap),
				m_heapOffset(other.m_heapOffset) {
			increment();
		}

		DenseStorage &operator=(const DenseStorage<T, d> &other) {
			if (this == &other) return *this;

			other.increment();
			decrement();
			m_refCount	 = other.m_refCount;
			m_size		 = other.m_size;
			m_heap		 = other.m_heap;
			m_heapOffset = other.m_heapOffset;

			return *this;
		}

		DenseStorage &operator=(DenseStorage<T, d> &&other) noexcept {
			if (this == &other) return *this;

			other.increment();
			decrement();
			m_refCount	 = other.m_refCount;
			m_size		 = other.m_size;
			m_heap		 = other.m_heap;
			m_heapOffset = other.m_heapOffset;

			return *this;
		}

		~DenseStorage() { decrement(); }

		operator bool() const { return m_refCount != nullptr; }

		ValueReference<T, d> operator[](i64 index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heapOffset + index);
		}

		ValueReference<T, d> operator[](i64 index) {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heapOffset + index);
		}

		void offsetMemory(i64 off) { m_heapOffset += off; }

		LR_NODISCARD("") LR_FORCE_INLINE T *__restrict heap() const {
			return m_heap;
		} // + m_memOffset; }

		LR_NODISCARD("") i64 size() const { return m_size; }

		LR_NODISCARD("") i64 bytes() const { return sizeof(T) * m_size; }

		LR_NODISCARD("") i64 getOffset() const { return std::distance(m_heapOffset, m_heap); }

		void setOffset(i64 off) { m_heapOffset = m_heap + off; }

	protected:
		void increment() const {
			if (!m_refCount) return;
			(*m_refCount)++;
		}

		void decrement() {
			if (!m_refCount) return;
			(*m_refCount)--;
			if (*m_refCount == 0) {
				delete m_refCount;
				memory::free<T, d>(m_heap);
			}
		}

		i64 m_size					 = 0;
		T *__restrict m_heap		 = nullptr;
		T *__restrict m_heapOffset	 = nullptr;
		std::atomic<i64> *m_refCount = nullptr;
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst, const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between DenseStorage objects with different sizes");
		memcpy<typename internal::traits<T>::BaseScalar,
			   d,
			   typename internal::traits<T_>::BaseScalar,
			   d_>(dst.heap(), src.heap(), dst.size());
		dst.setOffset(src.getOffset());
	}
} // namespace librapid::memory
