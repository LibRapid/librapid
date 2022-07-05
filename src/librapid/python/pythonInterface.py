import _librapid
from . import typeMapping
from . import extentInterface

print("HELLO WORLD!")

def test(x):
    print(x * 2)

def isArrayObject(obj):
	try:
		# CUDA support
		return isinstance(obj, (_librapid.ArrayB,
								_librapid.ArrayF16,
								_librapid.ArrayF32,
								_librapid.ArrayI32,
								_librapid.ArrayBG,
								_librapid.ArrayF16G,
								_librapid.ArrayF32G,
								_librapid.ArrayI32G))
	except:
		# No CUDA support
		return isinstance(obj, (_librapid.ArrayB,
								_librapid.ArrayC,
								_librapid.ArrayF16,
								_librapid.ArrayF32,
								_librapid.ArrayF64,
								_librapid.ArrayI16,
								_librapid.ArrayI32,
								_librapid.ArrayI64))

Extent = extentInterface.Extent

class Array:
	def __init__(self, *args, **kwargs):
		self._array = None
		type = None
		device = None
		extent = None

		for key, val in kwargs:
			if key in ("extent", "size", "dimensions"):
				if isinstance(val, Extent):
					extent = val
				elif isinstance(val, (list, tuple)):
					extent = Extent(val)
				else:
					raise ValueError("Extent argument must be a list-like object or Extent")
			elif key in ("type", "dtype", "datatype"):
				if isinstance(val, str):
					type = val
				else:
					raise ValueError("Type argument must be a string")
			elif key in ("device", "location", "accelerator"):
				if isinstance(val, str):
					device = val
				else:
					raise ValueError("Device argument must be a string")

		if len(args) > 0:
			if isArrayObject(args[0]):
				self._array = args[0]
			elif isinstance(args[0], Extent):
				extent = args[0]

				if len(args) > 1:
					type = args[1]
				if len(args) > 2:
					device = args[2]
				if len(args) > 3:
					raise ValueError("Array constructor expects <Extent, [type], [device]>")
			else:
				raise ValueError("Extent of Array is required")

		adjustedType, adjustedDevice = typeMapping.mapType(type, device)

		if adjustedType is None or adjustedDevice is None:
			raise ValueError("Array with type={} and device={} is invalid".format(type, device))

		if self._array is None:
			if adjustedDevice == "CPU":
				if adjustedType == "ArrayB":
					self._array = _librapid.ArrayB(extent)
				elif adjustedType == "ArrayC":
					self._array = _librapid.ArrayC(extent._extent)
				elif adjustedType == "ArrayF16":
					self._array = _librapid.ArrayF16(extent._extent)
				elif adjustedType == "ArrayF32":
					self._array = _librapid.ArrayF32(extent._extent)
				elif adjustedType == "ArrayF64":
					self._array = _librapid.ArrayF64(extent._extent)
				elif adjustedType == "ArrayI16":
					self._array = _librapid.ArrayI16(extent._extent)
				elif adjustedType == "ArrayI32":
					self._array = _librapid.ArrayI32(extent._extent)
				elif adjustedType == "ArrayI64":
					self._array = _librapid.ArrayI64(extent._extent)
			elif adjustedDevice == "GPU":
				if adjustedType == "ArrayB":
					self._array = _librapid.ArrayBG(extent._extent)
				elif adjustedType == "ArrayC":
					self._array = _librapid.ArrayCG(extent._extent)
				elif adjustedType == "ArrayF16":
					self._array = _librapid.ArrayF16G(extent._extent)
				elif adjustedType == "ArrayF32":
					self._array = _librapid.ArrayF32G(extent._extent)
				elif adjustedType == "ArrayF64":
					self._array = _librapid.ArrayF64G(extent._extent)
				elif adjustedType == "ArrayI16":
					self._array = _librapid.ArrayI16G(extent._extent)
				elif adjustedType == "ArrayI32":
					self._array = _librapid.ArrayI32G(extent._extent)
				elif adjustedType == "ArrayI64":
					self._array = _librapid.ArrayI64G(extent._extent)

	def copy(self):
		return Array(self._array.copy())

	def __getitem__(self, index:int):
		return Array(self._array[index])
	
	def __setitem__(self, index:int, val):
		self._array[index] = val

	def scalar(self, index:int):
		return self._array.scalar(index)

	def move(self, newLoc:str):
		_, adjusted = typeMapping.mapType(None, newLoc)
		if adjusted == "CPU":
			return Array(self._array.move_CPU())
		elif adjusted == "GPU":
			return Array(self._array.move_GPU())
		
		raise ValueError("Location \"{}\" is invalid".format(newLoc))

	def cast(self, newType:str):
		adjustedType, _ = typeMapping.mapType(newType, None)
		if adjustedType == "ArrayB":
			return Array(self._array.cast_ArrayB())
		elif adjustedType == "ArrayC":
			return Array(self._array.cast_ArrayC())
		elif adjustedType == "ArrayF16":
			return Array(self._array.cast_Array16F())
		elif adjustedType == "ArrayF32":
			return Array(self._array.cast_ArrayF32())
		elif adjustedType == "ArrayF64":
			return Array(self._array.cast_ArrayF64())
		elif adjustedType == "ArrayI16":
			return Array(self._array.cast_ArrayI16())
		elif adjustedType == "ArrayI32":
			return Array(self._array.cast_ArrayI32())
		elif adjustedType == "ArrayI64":
			return Array(self._array.cast_ArrayI64())

	def castMove(self, newType:str, newLoc:str):
		adjustedType, adjustedDevice = typeMapping.mapType(newType, newLoc)

		if adjustedDevice == "CPU":
			if adjustedType == "ArrayB":
				return Array(self._array.cast_ArrayB_CPU())
			elif adjustedType == "ArrayC":
				return Array(self._array.cast_ArrayC_CPU())
			elif adjustedType == "ArrayF16":
				return Array(self._array.cast_Array16F_CPU())
			elif adjustedType == "ArrayF32":
				return Array(self._array.cast_ArrayF32_CPU())
			elif adjustedType == "ArrayF64":
				return Array(self._array.cast_ArrayF64_CPU())
			elif adjustedType == "ArrayI16":
				return Array(self._array.cast_ArrayI16_CPU())
			elif adjustedType == "ArrayI32":
				return Array(self._array.cast_ArrayI32_CPU())
			elif adjustedType == "ArrayI64":
				return Array(self._array.cast_ArrayI64_CPU())
		elif adjustedDevice == "GPU":
			if adjustedType == "ArrayB":
				return Array(self._array.cast_ArrayB_GPU())
			elif adjustedType == "ArrayC":
				return Array(self._array.cast_ArrayC_GPU())
			elif adjustedType == "ArrayF16":
				return Array(self._array.cast_Array16F_GPU())
			elif adjustedType == "ArrayF32":
				return Array(self._array.cast_ArrayF32_GPU())
			elif adjustedType == "ArrayF64":
				return Array(self._array.cast_ArrayF64_GPU())
			elif adjustedType == "ArrayI16":
				return Array(self._array.cast_ArrayI16_GPU())
			elif adjustedType == "ArrayI32":
				return Array(self._array.cast_ArrayI32_GPU())
			elif adjustedType == "ArrayI64":
				return Array(self._array.cast_ArrayI64_GPU())

	def __add__(self, other):
		if isinstance(other, Array):
			return Array(self._array + other._array)
		return Array(self._array + other)

	def __sub__(self, other):
		if isinstance(other, Array):
			return Array(self._array - other._array)
		return Array(self._array - other)

	def __mul__(self, other):
		if isinstance(other, Array):
			return Array(self._array * other._array)
		return Array(self._array * other)

	def __div__(self, other):
		if isinstance(other, Array):
			return Array(self._array / other._array)
		return Array(self._array / other)

	def __or__(self, other):
		if isinstance(other, Array):
			return Array(self._array | other._array)
		return Array(self._array | other)

	def __and__(self, other):
		if isinstance(other, Array):
			return Array(self._array & other._array)
		return Array(self._array & other)

	def __xor__(self, other):
		if isinstance(other, Array):
			return Array(self._array ^ other._array)
		return Array(self._array ^ other)

	def __neg__(self):
		return Array(-self._array)

	def transpose(self, order:Extent=[]):
		self._array.transpose(order)

	def transposed(self, order:Extent=[]):
		return Array(self._array.transposed(order))

	def dot(self, other):
		return Array(self._array.dot(other._array))

	def str(self, format:str="{}",
				  delim:str=" ",
				  stripWidth:int=-1,
				  beforePoint:int=-1,
				  afterPoint:int=-1,
				  depth:int=0):
		return self._array.str(format,
							  delim,
							  stripWidth,
							  beforePoint,
							  afterPoint,
							  depth)

	def __str__(self):
		return self.str()

	def __repr__(self):
		return repr(self._array)

	def isScalar(self):
		return self._array.isScalar()

	def extent(self):
		return self._array.extent()

def add(lhs:Array, rhs:Array, dst:Array):
	_librapid.add(lhs._array, rhs._array, dst._array)

def sub(lhs:Array, rhs:Array, dst:Array):
	_librapid.sub(lhs._array, rhs._array, dst._array)

def mul(lhs:Array, rhs:Array, dst:Array):
	_librapid.mul(lhs._array, rhs.a_arrayrray, dst._array)

def div(lhs:Array, rhs:Array, dst:Array):
	_librapid.div(lhs._array, rhs._array, dst._array)
