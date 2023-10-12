#ifndef LIBRAPID_SET_HPP
#define LIBRAPID_SET_HPP

namespace librapid {
	/// \brief An unordered set of distinct elements of the same type. Elements are stored in
	/// ascending order and duplicates are removed.
	///
	/// For example, consider creating a set from the following array:
	/// \code
	/// myArr = { 4, 4, 3, 7, 5, 2, 5, 6, 7, 1, 8, 9 }
	/// mySet = Set(myArr)
	/// // mySet -> Set(1, 2, 3, 4, 5, 6, 7, 8, 9)
	/// \endcode
	/// \tparam ElementType_ The type of the elements in the set
	template<typename ElementType_>
	class Set {
	public:
		/// \brief The type of the elements in the set
		using ElementType = ElementType_;

		/// \brief The type of the underlying vector
		using VectorType = std::vector<ElementType>;

		/// \brief The type of the iterator for the underlying vector
		using VectorIterator = typename VectorType::iterator;

		/// \brief The type of the const iterator for the underlying vector
		using VectorConstIterator = typename VectorType::const_iterator;

		/// \brief Default constructor
		Set() = default;

		/// \brief Copy constructor
		Set(const Set &other) = default;

		/// \brief Move constructor
		Set(Set &&other) = default;

		/// \brief Construct a set from an array
		///
		/// In the case of multidimensional arrays, the elements are flattened. This means a 2D
		/// array will still result in a 1D set. Create a ``Set<Set<...>>`` if needed.
		/// \tparam ShapeType Shape type of the array
		/// \tparam StorageType Storage type of the array
		/// \param arr The array to construct the set from
		template<typename ShapeType, typename StorageType>
		Set(const array::ArrayContainer<ShapeType, StorageType> &arr) {
			reserve(arr.size());
			for (size_t i = 0; i < arr.size(); ++i) { pushBack(arr.storage()[i]); }
			sort();
			prune();
		}

		/// \brief Construct a set from a vector
		/// \param data The vector to construct the set from
		Set(const std::vector<ElementType> &data) : m_data(data) {
			sort();
			prune();
		}

		/// \brief Construct a set from an initializer list
		/// \param data The initializer list to construct the set from
		Set(const std::initializer_list<ElementType> &data) : m_data(data) {
			sort();
			prune();
		}

		/// \brief Copy assignment operator
		Set &operator=(const Set &other) = default;

		/// \brief Move assignment operator
		Set &operator=(Set &&other) = default;

		/// \return Return the cardinality of the set \f$ |S| \f$
		LIBRAPID_NODISCARD int64_t size() const { return m_data.size(); }

		/// \brief Access the n-th element of the set in ascending order (i.e. \f$ S_n \f$)
		///
		/// \p index must be in the range \f$ [0, |S|) \f$
		/// \param index The index of the element to access
		/// \return Return the n-th element of the set
		LIBRAPID_NODISCARD const ElementType &operator[](int64_t index) const {
			LIBRAPID_ASSERT(index >= 0 && index < m_data.size(),
							"Index out of bounds: {} (size: {})",
							index,
							m_data.size());
			return m_data[index];
		}

		/// \brief Check if the set contains a value (\f$ \text{val} \in S \f$)
		/// \param val The value to check for
		/// \return True if the value is present, false otherwise
		LIBRAPID_NODISCARD bool contains(const ElementType &val) const {
			// Binary search

			int64_t mid;
			int64_t head = 0;
			int64_t tail = m_data.size() - 1;
			bool found	 = false;

			while (!found && head <= tail) {
				mid = (head + tail) / 2;
				if (val < m_data[mid]) {
					tail = mid - 1;
				} else if (val > m_data[mid]) {
					head = mid + 1;
				} else {
					found = true;
				}
			}

			return found;
		}

		/// \brief Insert a value into the set (\f$ S \cup \{\text{val}\} \f$)
		/// \param val The value to insert
		/// \return Return a reference to the set
		Set &insert(const ElementType &val) {
			if (contains(val)) return *this;

			// Insert into data
			for (auto it = m_data.begin(); it < m_data.end(); it++) {
				if (*it > val) {
					m_data.insert(it, val);
					return *this;
				}
			}

			m_data.emplace_back(val);
			return *this;
		}

		/// \brief Insert an `std::vector` of values into the set (\f$ S \leftarrow S \cup
		/// \text{data} \f$)
		///
		/// Each element of the vector is inserted into the set.
		/// \param data
		/// \return Reference to the set
		Set &insert(const std::vector<ElementType> &data) {
			for (const auto &val : data) { insert(val); }
			return *this;
		}

		/// \brief Insert an initializer list of values into the set (\f$ S \leftarrow S \cup
		/// \text{data} \f$) \param data \return Reference to the set
		Set &insert(const std::initializer_list<ElementType> &data) {
			for (const auto &val : data) { insert(val); }
			return *this;
		}

		/// \brief Insert an element into the set (\f$ S \leftarrow S \cup \{\text{val}\} \f$)
		/// \param val The value to insert
		/// \return Return a reference to the set
		/// \see insert(const ElementType &val)
		Set &operator+=(const ElementType &val) { return insert(val); }

		/// \brief Insert an `std::vector` of values into the set (\f$ S \leftarrow S \cup
		/// \text{data} \f$) \param data \return Reference to the set \see insert(const
		/// std::vector<ElementType> &data)
		Set &operator+=(const std::vector<ElementType> &data) { return insert(data); }

		/// \brief Insert an initializer list of values into the set (\f$ S \leftarrow S \cup
		/// \text{data} \f$) \param data \return Reference to the set \see insert(const
		/// std::initializer_list<ElementType> &data)
		Set &operator+=(const std::initializer_list<ElementType> &data) { return insert(data); }

		/// \brief Insert an element into the set and return the result
		/// (\f$ R = S \cup \{\text{val}\} \f$)
		/// \param val The value to insert
		/// \return A new set \f$ R = S \cup \{\text{val}\} \f$
		Set operator+(const ElementType &val) {
			Set result = *this;
			return result += val;
		}

		/// \brief Insert an `std::vector` of values into the set and return the result
		/// (\f$ R = S \cup \text{data} \f$)
		/// \param data The vector of values to insert
		/// \return A new set \f$ R = S \cup \text{data} \f$
		Set operator+(const std::vector<ElementType> &data) {
			Set result = *this;
			return result += data;
		}

		/// \brief Insert an initializer list of values into the set and return the result
		/// (\f$ R = S \cup \text{data} \f$)
		/// \param data The initializer list of values to insert
		/// \return A new set \f$ R = S \cup \text{data} \f$
		Set operator+(const std::initializer_list<ElementType> &data) {
			Set result = *this;
			return result += data;
		}

		/// \brief Discard \p val from the set if it exists (\f$ S \setminus \{\text{val}\} \f$)
		///
		/// If \p val is not contained within the set, nothing happens.
		///
		/// \param val The value to discard
		/// \return A reference to the set
		Set &discard(const ElementType &val) {
			// Binary search

			int64_t mid;
			int64_t head = 0;
			int64_t tail = m_data.size() - 1;

			while (head <= tail) {
				mid = (head + tail) / 2;
				if (val < m_data[mid]) {
					tail = mid - 1;
				} else if (val > m_data[mid]) {
					head = mid + 1;
				} else {
					m_data.erase(m_data.begin() + mid);
					return *this;
				}
			}

			return *this;
		}

		/// \brief Discard an `std::vector` of values from the set (\f$ S \setminus \text{data} \f$)
		///
		/// If an element in \p data is not contained within the set, nothing happens.
		///
		/// \param data The vector of values to Discard
		/// \return A reference to the set
		Set &discard(const std::vector<ElementType> &data) {
			for (const auto &val : data) { discard(val); }
			return *this;
		}

		/// \brief Discard an initializer list of values from the set (\f$ S \setminus \text{data}
		/// \f$)
		///
		/// If an element in \p data is not contained within the set, nothing happens.
		///
		/// \param data The initializer list of values to Discard
		/// \return A reference to the set
		Set &discard(const std::initializer_list<ElementType> &data) {
			for (const auto &val : data) { discard(val); }
			return *this;
		}

		/// \brief Remove \p val from the set (\f$ S \setminus \{\text{val}\} \f$)
		///
		/// If \p val is not contained within the set, an exception is thrown.
		///
		/// \param val The value to remove
		/// \return A reference to the set
		Set &remove(const ElementType &val) {
			LIBRAPID_ASSERT(contains(val), "Set does not contain value: {}", val);
			return discard(val);
		}

		/// \brief Remove an `std::vector` of values from the set (\f$ S \setminus \text{data} \f$)
		///
		/// If an element in \p data is not contained within the set, an exception is thrown.
		///
		/// \param data The vector of values to remove
		/// \return A reference to the set
		Set &remove(const std::vector<ElementType> &data) {
			for (const auto &val : data) { remove(val); }
			return *this;
		}

		/// \brief Remove an initializer list of values from the set (\f$ S \setminus \text{data}
		/// \f$)
		///
		/// If an element in \p data is not contained within the set, an exception is thrown.
		///
		/// \param data The initializer list of values to remove
		/// \return  A reference to the set
		Set &remove(const std::initializer_list<ElementType> &data) {
			for (const auto &val : data) { remove(val); }
			return *this;
		}

		/// \brief Discard \p val from the set if it exists
		/// \param val The value to discard
		/// \return A reference to the set
		/// \see discard(const ElementType &val)
		Set &operator-=(const ElementType &val) { return discard(val); }

		/// \brief Discard an `std::vector` of values from the set
		/// \param data The vector of values to discard
		/// \return A reference to the set
		/// \see discard(const std::vector<ElementType> &data)
		Set &operator-=(const std::vector<ElementType> &data) { return discard(data); }

		/// \brief Discard an initializer list of values from the set
		/// \param data The initializer list of values to discard
		/// \return A reference to the set
		/// \see discard(const std::initializer_list<ElementType> &data)
		Set &operator-=(const std::initializer_list<ElementType> &data) { return discard(data); }

		/// \brief Discard \p val from the set if it exists and return the result
		/// \param val The value to discard
		/// \return A new set \f$ R = S \setminus \{\text{val}\} \f$
		/// \see discard(const ElementType &val)
		Set operator-(const ElementType &val) {
			Set result = *this;
			return result -= val;
		}

		/// \brief Discard an `std::vector` of values from the set and return the result
		/// \param data The vector of values to discard
		/// \return A new set \f$ R = S \setminus \text{data} \f$
		/// \see discard(const std::vector<ElementType> &data)
		Set operator-(const std::vector<ElementType> &data) {
			Set result = *this;
			return result -= data;
		}

		/// \brief Discard an initializer list of values from the set and return the result
		/// \param data The initializer list of values to discard
		/// \return A new set \f$ R = S \setminus \text{data} \f$
		/// \see discard(const std::initializer_list<ElementType> &data)
		Set operator-(const std::initializer_list<ElementType> &data) {
			Set result = *this;
			return result -= data;
		}

		/// \brief Return the union of two sets (\f$ R = S_1 \cup S_2 \f$)
		///
		/// \f$ \{ x : x \in S_1 \lor x \in S_2 \} \f$
		///
		/// \param other \f$ S_2 \f$
		/// \return A new set \f$ R = S_1 \cup S_2 \f$
		LIBRAPID_NODISCARD Set operator|(const Set &other) const {
			// Union operator
			Set result;

			// Reserve space for elements (a decent guess)
			result.reserve(m_data.size() + other.m_data.size());

			int64_t indexA = 0;
			int64_t indexB = 0;

			while (indexA < m_data.size() && indexB < other.m_data.size()) {
				if (m_data[indexA] < other.m_data[indexB]) {
					result.pushBack(m_data[indexA]);
					++indexA;
				} else if (m_data[indexA] > other.m_data[indexB]) {
					result.pushBack(other.m_data[indexB]);
					++indexB;
				} else {
					result.pushBack(m_data[indexA]);
					++indexA;
					++indexB;
				}
			}

			// Add remaining elements
			result.insert(result.end(), m_data.begin() + indexA, m_data.end());
			result.insert(result.end(), other.m_data.begin() + indexB, other.m_data.end());

			return result;
		}

		/// \brief Return the intersection of two sets (\f$ R = S_1 \cap S_2 \f$)
		///
		/// \f$ \{ x : x \in S_1 \land x \in S_2 \} \f$
		///
		/// \param other \f$ S_2 \f$
		/// \return A new set \f$ R = S_1 \cap S_2 \f$
		Set operator&(const Set &other) const {
			// Intersection operator
			Set result;

			// Reserve space for elements (a decent guess)
			result.reserve(std::min(m_data.size(), other.m_data.size()));

			int64_t indexA = 0;
			int64_t indexB = 0;

			while (indexA < m_data.size() && indexB < other.m_data.size()) {
				if (m_data[indexA] < other.m_data[indexB]) {
					++indexA;
				} else if (m_data[indexA] > other.m_data[indexB]) {
					++indexB;
				} else {
					result.pushBack(m_data[indexA]);
					++indexA;
					++indexB;
				}
			}

			return result;
		}

		/// \brief Return the symmetric difference of two sets (\f$ R = S_1 \oplus S_2 \f$)
		///
		/// \f$ \{ x : x \in S_1 \oplus x \in S_2 \} \f$
		///
		/// \param other \f$ S_2 \f$
		/// \return A new set \f$ R = S_1 \oplus S_2 \f$
		Set operator^(const Set &other) const {
			// Symmetric difference operator (elements in either set but not both)
			Set result;

			// Reserve space for elements (a decent guess)
			result.reserve(m_data.size() + other.m_data.size());

			int64_t indexA = 0;
			int64_t indexB = 0;

			while (indexA < m_data.size() && indexB < other.m_data.size()) {
				if (m_data[indexA] < other.m_data[indexB]) {
					result.pushBack(m_data[indexA]);
					++indexA;
				} else if (m_data[indexA] > other.m_data[indexB]) {
					result.pushBack(other.m_data[indexB]);
					++indexB;
				} else {
					++indexA;
					++indexB;
				}
			}

			return result;
		}

		/// \brief Return the set difference of two sets (\f$ R = S_1 \setminus S_2 \f$)
		///
		/// \f$ \{ x : x \in S_1 \land x \notin S_2 \} \f$
		///
		/// \param other \f$ S_2 \f$
		/// \return A new set \f$ R = S_1 \setminus S_2 \f$
		Set operator-(const Set &other) const {
			// Set difference
			Set result;

			// Reserve space for elements (a decent guess)
			result.reserve(m_data.size() + other.m_data.size());

			int64_t indexA = 0;
			int64_t indexB = 0;

			while (indexA < m_data.size() && indexB < other.m_data.size()) {
				if (m_data[indexA] < other.m_data[indexB]) {
					result.pushBack(m_data[indexA]);
					++indexA;
				} else if (m_data[indexA] > other.m_data[indexB]) {
					++indexB;
				} else {
					++indexA;
					++indexB;
				}
			}

			// Add remaining elements
			result.insert(result.end(), m_data.begin() + indexA, m_data.end());

			return result;
		}

		LIBRAPID_NODISCARD auto operator<=>(const Set &other) const = default;

		/// \return Iterator to the beginning of the set
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto begin() const { return m_data.begin(); }

		/// \return Iterator to the end of the set
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto end() const { return m_data.end(); }

		/// \brief Used for formatting the set with {fmt}
		/// \tparam T Formatter type
		/// \tparam Char Character type
		/// \tparam Ctx Context type
		/// \param format ``formatter`` instance
		/// \param ctx ``format_context`` instance
		template<typename T, typename Char, typename Ctx>
		void str(const fmt::formatter<T, Char> &format, Ctx &ctx) const {
			fmt::format_to(ctx.out(), "(");
			for (int64_t i = 0; i < m_data.size(); ++i) {
				format.format(m_data[i], ctx);
				if (i < m_data.size() - 1) { fmt::format_to(ctx.out(), ", "); }
			}
			fmt::format_to(ctx.out(), ")");
		}

	protected:
		/// \brief Reserve space in the underlying vector for \p elements elements
		/// \param elements The number of elements to reserve space for
		void reserve(size_t elements) { m_data.reserve(elements); }

		/// \brief Sort the underlying vector
		void sort() { std::sort(m_data.begin(), m_data.end()); }

		/// \brief Remove duplicates from the underlying vector
		void prune() {
			std::vector<ElementType> newData;
			newData.reserve(m_data.size());
			for (int64_t i = 0; i < m_data.size(); ++i) {
				if (i == 0 || m_data[i] != m_data[i - 1]) { newData.emplace_back(m_data[i]); }
			}
			m_data = newData;
		}

		/// \brief Add a value to the end of the set if it is known to be the largest element
		/// \param val
		LIBRAPID_ALWAYS_INLINE void pushBack(const ElementType &val) { m_data.emplace_back(val); }

		/// \brief Insert a vector of values into a location in the set if they are known to
		/// be valid in that position
		/// \param insertLocation
		/// \param begin
		/// \param end
		LIBRAPID_ALWAYS_INLINE void insert(VectorConstIterator insertLocation,
										   VectorConstIterator begin, VectorConstIterator end) {
			m_data.insert(insertLocation, begin, end);
		}

	private:
		/// \brief The underlying vector
		std::vector<ElementType> m_data;
	};
} // namespace librapid

template<typename ElementType, typename Char>
struct fmt::formatter<librapid::Set<ElementType>, Char> {
private:
	using Type = librapid::Set<ElementType>;
	using Base = fmt::formatter<ElementType, Char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const Char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const Type &val, FormatContext &ctx) const -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};

LIBRAPID_SIMPLE_IO_NORANGE(typename ElementType, librapid::Set<ElementType>)

// std ostream support
template<typename ElementType>
std::ostream &operator<<(std::ostream &os, const librapid::Set<ElementType> &set) {
	return os << fmt::format("{}", set);
}

#endif // LIBRAPID_SET_HPP