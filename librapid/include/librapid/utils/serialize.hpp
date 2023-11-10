#ifndef LIBRAPID_UTILS_SERIALIZE_HPP
#define LIBRAPID_UTILS_SERIALIZE_HPP

#include <fstream>

namespace librapid::serialize {
	namespace detail {
		inline std::ios::openmode fileBinMode(const std::string &path) {
			if (path.find(".bin") != std::string::npos) {
				return std::ios::out | std::ios::binary;
			} else {
				return std::ios::out;
			}
		}
	} // namespace detail

	template<typename T>
	struct SerializerImpl {
		// Used to ensure that the type is the same when deserializing
		LIBRAPID_NODISCARD static size_t hasher() {
			auto type	= std::string(typetraits::typeName<T>());
			size_t hash = std::hash<std::string> {}(type);
			return hash;
		}

		LIBRAPID_NODISCARD static std::vector<uint8_t> serialize(const T &obj) {
			std::vector<uint8_t> data;
			data.resize(sizeof(T) + sizeof(size_t));
			size_t hashed = hasher();
			std::memcpy(data.data() + sizeof(T), &hashed, sizeof(size_t));
			std::memcpy(data.data(), &obj, sizeof(T));
			return data;
		}

		LIBRAPID_NODISCARD static T deserialize(const std::vector<uint8_t> &data) {
			size_t hashed;
			std::memcpy(&hashed, data.data() + sizeof(T), sizeof(size_t));
			LIBRAPID_ASSERT(
			  hasher() == hashed,
			  "Hash mismatch. Ensure the types are the same and the data is not corrupted.");

			T obj;
			std::memcpy(&obj, data.data(), sizeof(T));
			return obj;
		}
	};

	template<typename T>
	class Serializer {
	public:
		Serializer() = default;

		explicit Serializer(const T &obj) : m_data(SerializerImpl<T>::serialize(obj)) {}

		Serializer(const Serializer<T> &other) = default;
		Serializer(Serializer<T> &&other)	   = default;

		Serializer<T> &operator=(const Serializer<T> &other) = default;
		Serializer<T> &operator=(Serializer<T> &&other)		 = default;

		~Serializer() = default;

		LIBRAPID_NODISCARD const std::vector<uint8_t> &data() const { return m_data; }
		LIBRAPID_NODISCARD std::vector<uint8_t> &data() { return m_data; }

		void serialize(const T &obj) { m_data = SerializerImpl<T>::serialize(obj); }
		LIBRAPID_NODISCARD T deserialize() const { return SerializerImpl<T>::deserialize(m_data); }

		LIBRAPID_NODISCARD bool write(std::fstream &file) const {
			file.write(reinterpret_cast<const char *>(m_data.data()), m_data.size());
			return file.good();
		}

		LIBRAPID_NODISCARD bool write(const std::string &path) const {
			std::fstream file(path, std::ios::out | detail::fileBinMode(path));
			return write(file);
		}

		LIBRAPID_NODISCARD bool read(std::fstream &file) {
			file.seekg(0, std::ios::end);
			m_data.resize(file.tellg());
			file.seekg(0, std::ios::beg);
			file.read(reinterpret_cast<char *>(m_data.data()), m_data.size());
			return file.good();
		}

		LIBRAPID_NODISCARD bool read(const std::string &path) {
			std::fstream f(path, std::ios::in | detail::fileBinMode(path));
			return read(f);
		}

	private:
		std::vector<uint8_t> m_data;
	};
} // namespace librapid::serialize

#endif // LIBRAPID_UTILS_SERIALIZE_HPP