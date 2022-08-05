import _librapid
from . import typeMapping
from . import extentInterface

def test(x):
    print(x * 2)

def isArrayObject(obj):
	try:
		# CUDA support
		return isinstance(obj, (_librapid.ArrayB,
								_librapid.ArrayC,
								_librapid.ArrayF16,
								_librapid.ArrayF32,
								_librapid.ArrayF64,
								_librapid.ArrayI16,
								_librapid.ArrayI32,
								_librapid.ArrayI64,
								_librapid.ArrayMPZ,
								_librapid.ArrayMPQ,
								_librapid.ArrayMPFR,
								_librapid.ArrayCF32,
								_librapid.ArrayCF64,
								_librapid.ArrayCMPFR,
								_librapid.ArrayBG,
								_librapid.ArrayCG,
								_librapid.ArrayF16G,
								_librapid.ArrayF32G,
								_librapid.ArrayF64G,
								_librapid.ArrayI16G,
								_librapid.ArrayI32G,
								_librapid.ArrayI64G))
	except:
		# No CUDA support
		return isinstance(obj, (_librapid.ArrayB,
								_librapid.ArrayC,
								_librapid.ArrayF16,
								_librapid.ArrayF32,
								_librapid.ArrayF64,
								_librapid.ArrayI16,
								_librapid.ArrayI32,
								_librapid.ArrayI64,
								_librapid.ArrayMPZ,
								_librapid.ArrayMPQ,
								_librapid.ArrayMPFR,
								_librapid.ArrayCF32,
								_librapid.ArrayCF64,
								_librapid.ArrayCMPFR))

Extent = extentInterface.Extent

mpz = _librapid.mpz
mpf = _librapid.mpf
mpq = _librapid.mpq
mpfr = _librapid.mpfr

ComplexF32 = _librapid.ComplexF32
ComplexF64 = _librapid.ComplexF64
ComplexMPFR = _librapid.ComplexMPFR

prec = _librapid.prec

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
				elif adjustedType == "ArrayMPZ":
					self._array = _librapid.ArrayMPZ(extent._extent)
				elif adjustedType == "ArrayMPQ":
					self._array = _librapid.ArrayMPQ(extent._extent)
				elif adjustedType == "ArrayMPFR":
					self._array = _librapid.ArrayMPFR(extent._extent)
				elif adjustedType == "ArrayCF32":
					self._array = _librapid.ArrayCF32(extent._extent)
				elif adjustedType == "ArrayCF64":
					self._array = _librapid.ArrayCF64(extent._extent)
				elif adjustedType == "ArrayCMPFR":
					self._array = _librapid.ArrayCMPFR(extent._extent)
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

		if self._array is None:
			raise ValueError("Unknown or invalid Array type")

	def copy(self):
		return Array(self._array.copy())

	def __getitem__(self, index:int):
		return Array(self._array[index])
	
	def __setitem__(self, index:int, val):
		self._array[index] = val

	def __call__(self, *args):
		return self._array(*args)

	def get(self, *args):
		return self._array.get(*args)

	def set(self, val, *args):
		self._array.set(val, *args)

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
		elif adjustedType == "ArrayMPZ":
			return Array(self._array.cast_ArrayMPZ())
		elif adjustedType == "ArrayMPQ":
			return Array(self._array.cast_ArrayMPQ())
		elif adjustedType == "ArrayMPFR":
			return Array(self._array.cast_ArrayMPFR())
		elif adjustedType == "ArrayCF32":
			return Array(self._array.cast_ArrayCF32())
		elif adjustedType == "ArrayCF64":
			return Array(self._array.cast_ArrayCF64())
		elif adjustedType == "ArrayCMPFR":
			return Array(self._array.cast_ArrayCMPFR())

	def castMove(self, newType:str, newLoc:str):
		adjustedType, adjustedDevice = typeMapping.mapType(newType, newLoc)

		return self.cast(adjustedType).move(adjustedDevice)

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

# Functions from the Math library
abs = _librapid.abs
floor = _librapid.floor
ceil = _librapid.ceil
pow = _librapid.pow
sqrt = _librapid.sqrt
exp = _librapid.exp
exp2 = _librapid.exp2
exp10 = _librapid.exp10
log = _librapid.log
log2 = _librapid.log2
log10 = _librapid.log10
sin = _librapid.sin
cos = _librapid.cos
tan = _librapid.tan
asin = _librapid.asin
acos = _librapid.acos
atan = _librapid.atan
csc = _librapid.csc
sec = _librapid.sec
cot = _librapid.cot
acsc = _librapid.acsc
asec = _librapid.asec
acot = _librapid.acot
sinh = _librapid.sinh
cosh = _librapid.cosh
tanh = _librapid.tanh
asinh = _librapid.asinh
acosh = _librapid.acosh
atanh = _librapid.atanh
mod = _librapid.mod
round = _librapid.round
roundSigFig = _librapid.roundSigFig
roundTo = _librapid.roundTo
roundUpTo = _librapid.roundUpTo
map = _librapid.map
random = _librapid.random
randint = _librapid.randint
trueRandomEntropy = _librapid.trueRandomEntropy
trueRandom = _librapid.trueRandom
trueRandint = _librapid.trueRandint
randomGaussian = _librapid.randomGaussian
pow10 = _librapid.pow10
lerp = _librapid.lerp

constPi = _librapid.constPi
constEuler = _librapid.constEuler
constLog2 = _librapid.constLog2
constCatalan = _librapid.constCatalan

norm = _librapid.norm
polar = _librapid.polar
