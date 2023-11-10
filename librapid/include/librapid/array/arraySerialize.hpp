#ifndef LIBRAPID_ARRAY_SERIALIZE
#define LIBRAPID_ARRAY_SERIALIZE

namespace librapid::serialize {
	template<>
	struct SerializerImpl<Shape> {
		LIBRAPID_NODISCARD static size_t hasher() {
			return std::hash<std::string> {}(std::string("Shape"));
		}

		LIBRAPID_NODISCARD static std::vector<uint8_t> serialize(const Shape &shape) {
			// Serialized structure:
			// - Number of dimensions
			// - Shape (padded to LIBRAPID_MAX_ARRAY_DIMS)

			using SizeType = Shape::SizeType;
			using DimType  = Shape::DimType;

			std::vector<uint8_t> serialized;
			serialized.resize(sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS + sizeof(DimType) +
							  sizeof(size_t));

			DimType numDims = shape.ndim();
			size_t hashed	= hasher();
			std::memcpy(serialized.data(), &numDims, sizeof(DimType));
			std::memcpy(serialized.data() + sizeof(DimType),
						shape.data().begin(),
						sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS);
			std::memcpy(serialized.data() + sizeof(DimType) +
						  sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS,
						&hashed,
						sizeof(size_t));
			return serialized;
		}

		LIBRAPID_NODISCARD static Shape deserialize(const std::vector<uint8_t> &data) {
			// Serialized structure:
			// - Number of dimensions
			// - Shape (padded to LIBRAPID_MAX_ARRAY_DIMS)

			using SizeType = Shape::SizeType;
			using DimType  = Shape::DimType;

			DimType numDims;
			size_t hashed;
			std::memcpy(&numDims, data.data(), sizeof(DimType));
			Shape shape = Shape::zeros(numDims);
			std::memcpy(shape.data().begin(),
						data.data() + sizeof(DimType),
						sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS);
			std::memcpy(&hashed,
						data.data() + sizeof(DimType) + sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS,
						sizeof(size_t));

			LIBRAPID_ASSERT(
			  hashed == hasher(),
			  "Hash mismatch. Ensure the types are the same and the data is not corrupted.");

			return shape;
		}
	};

	template<typename Scalar>
	struct SerializerImpl<Storage<Scalar>> {
		LIBRAPID_NODISCARD static size_t hasher() {
			return std::hash<std::string> {}(
			  fmt::format("Storage{}", typetraits::typeName<Scalar>()));
		}

		LIBRAPID_NODISCARD static std::vector<uint8_t> serialize(const Storage<Scalar> &storage) {
			// Serialized structure:
			// - Number of elements
			// - Data
			// - Hash

			std::vector<uint8_t> serialized;
			serialized.resize(sizeof(size_t) * 2 + sizeof(Scalar) * storage.size());

			size_t elements = storage.size();
			size_t hashed	= hasher();
			std::memcpy(serialized.data(), &elements, sizeof(size_t));
			std::memcpy(
			  serialized.data() + sizeof(size_t), storage.data(), sizeof(Scalar) * storage.size());
			std::memcpy(serialized.data() + sizeof(size_t) + sizeof(Scalar) * storage.size(),
						&hashed,
						sizeof(size_t));
			return serialized;
		}

		LIBRAPID_NODISCARD static Storage<Scalar> deserialize(const std::vector<uint8_t> &data) {
			// Serialized structure:
			// - Number of elements
			// - Data
			// - Hash

			size_t elements;
			size_t hashed;
			std::memcpy(&elements, data.data(), sizeof(size_t));
			Storage<Scalar> storage(elements);
			std::memcpy(
			  storage.data(), data.data() + sizeof(size_t), sizeof(Scalar) * storage.size());
			std::memcpy(&hashed,
						data.data() + sizeof(size_t) + sizeof(Scalar) * storage.size(),
						sizeof(size_t));

			LIBRAPID_ASSERT(
			  hashed == hasher(),
			  "Hash mismatch. Ensure the types are the same and the data is not corrupted.");

			return storage;
		}
	};

	template<typename ShapeType, typename StorageType>
	struct SerializerImpl<array::ArrayContainer<ShapeType, StorageType>> {
		using Type = array::ArrayContainer<ShapeType, StorageType>;

		LIBRAPID_NODISCARD static size_t hasher() {
			// Hash data:
			// - Data type
			// - Shape Type
			// - Storage Type (encodes backend as well)

			using Scalar = typename typetraits::TypeInfo<Type>::Scalar;

			size_t hash = 0;
			hash ^= SerializerImpl<Scalar>::hasher();
			hash ^= SerializerImpl<ShapeType>::hasher();
			hash ^= SerializerImpl<StorageType>::hasher();
			return hash;
		}

		LIBRAPID_NODISCARD static std::vector<uint8_t> serialize(const Type &arr) {
			// Serialized structure:
			// - Shape
			// - Data

			using Scalar   = typename typetraits::TypeInfo<Type>::Scalar;
			using SizeType = typename ShapeType::SizeType;
			using DimType  = typename ShapeType::DimType;

			const ShapeType &shape	   = arr.shape();
			const StorageType &storage = arr.storage();
			size_t hashed			   = hasher();

			size_t shapeBytes =
			  sizeof(SizeType) * LIBRAPID_MAX_ARRAY_DIMS + sizeof(DimType) + sizeof(size_t);
			size_t storageBytes = sizeof(Scalar) * storage.size() + sizeof(size_t) * 2;

			std::vector<uint8_t> serialized;
			serialized.resize(shapeBytes + storageBytes);

			std::memcpy(
			  serialized.data(), SerializerImpl<ShapeType>::serialize(shape).data(), shapeBytes);
			std::memcpy(serialized.data() + shapeBytes,
						SerializerImpl<StorageType>::serialize(storage).data(),
						storageBytes);
			std::memcpy(serialized.data() + shapeBytes + storageBytes, &hashed, sizeof(size_t));

			return serialized;
		}

		LIBRAPID_NODISCARD static Type deserialize(const std::vector<uint8_t> &data) {
			// Serialized structure:
			// - Shape
			// - Data

			using Scalar = typename typetraits::TypeInfo<Type>::Scalar;

			ShapeType shape		= SerializerImpl<ShapeType>::deserialize(data);
			StorageType storage = SerializerImpl<StorageType>::deserialize(std::vector<uint8_t>(
			  data.begin() + SerializerImpl<ShapeType>::serialize(shape).size(), data.end()));

			Type ret;
			ret.size_()	  = shape.size();
			ret.shape()	  = shape;
			ret.storage() = storage;
			return ret;
		}
	};
} // namespace librapid::serialize

#endif // LIBRAPID_ARRAY_SERIALIZE