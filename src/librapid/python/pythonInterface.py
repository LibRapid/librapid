import inspect
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

half = _librapid.half
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

	def __truediv__(self, other):
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

	# ========= Reverse Operators =========

	def __radd__(self, other):
		if isinstance(other, Array):
			return Array(self._array + other._array)
		return Array(other + self._array)

	def __rsub__(self, other):
		if isinstance(other, Array):
			return Array(self._array - other._array)
		return Array(other - self._array)

	def __rmul__(self, other):
		if isinstance(other, Array):
			return Array(self._array * other._array)
		return Array(other * self._array)

	def __rtruediv__(self, other):
		if isinstance(other, Array):
			return Array(self._array / other._array)
		return Array(other / self._array)

	def __ror__(self, other):
		if isinstance(other, Array):
			return Array(self._array | other._array)
		return Array(other | self._array)

	def __rand__(self, other):
		if isinstance(other, Array):
			return Array(self._array & other._array)
		return Array(other & self._array)

	def __rxor__(self, other):
		if isinstance(other, Array):
			return Array(self._array ^ other._array)
		return Array(other ^ self._array)

	def fill(self, val):
		self._array.fill(val)

	def filled(self, val):
		return Array(self._array.filled(val))

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

# Utility Functions
setNumThreads = _librapid.setNumThreads
setBlasThreads = _librapid.setBlasThreads
setCudaMathMode = _librapid.setCudaMathMode

def _generateFunction(function):
	def newFunction(*args):
		newArgs = (arg._array if isinstance(arg, Array) else arg for arg in args)
		containsArray = any([isinstance(arg, Array) for arg in args])
		res = function(*newArgs)
		if res is not None and containsArray:
			return Array(res)
		return res

	return newFunction

# Functions from the Math library
abs = _generateFunction(_librapid.abs)
floor = _generateFunction(_librapid.floor)
ceil = _generateFunction(_librapid.ceil)
pow = _generateFunction(_librapid.pow)
sqrt = _generateFunction(_librapid.sqrt)
exp = _generateFunction(_librapid.exp)
exp2 = _generateFunction(_librapid.exp2)
exp10 = _generateFunction(_librapid.exp10)
log = _generateFunction(_librapid.log)
log2 = _generateFunction(_librapid.log2)
log10 = _generateFunction(_librapid.log10)
sin = _generateFunction(_librapid.sin)
cos = _generateFunction(_librapid.cos)
tan = _generateFunction(_librapid.tan)
asin = _generateFunction(_librapid.asin)
acos = _generateFunction(_librapid.acos)
atan = _generateFunction(_librapid.atan)
csc = _generateFunction(_librapid.csc)
sec = _generateFunction(_librapid.sec)
cot = _generateFunction(_librapid.cot)
acsc = _generateFunction(_librapid.acsc)
asec = _generateFunction(_librapid.asec)
acot = _generateFunction(_librapid.acot)
sinh = _generateFunction(_librapid.sinh)
cosh = _generateFunction(_librapid.cosh)
tanh = _generateFunction(_librapid.tanh)
asinh = _generateFunction(_librapid.asinh)
acosh = _generateFunction(_librapid.acosh)
atanh = _generateFunction(_librapid.atanh)
mod = _generateFunction(_librapid.mod)
round = _generateFunction(_librapid.round)
roundSigFig = _generateFunction(_librapid.roundSigFig)
roundTo = _generateFunction(_librapid.roundTo)
roundUpTo = _generateFunction(_librapid.roundUpTo)
map = _generateFunction(_librapid.map)
random = _generateFunction(_librapid.random)
randint = _generateFunction(_librapid.randint)
trueRandomEntropy = _generateFunction(_librapid.trueRandomEntropy)
trueRandom = _generateFunction(_librapid.trueRandom)
trueRandint = _generateFunction(_librapid.trueRandint)
randomGaussian = _generateFunction(_librapid.randomGaussian)
pow10 = _generateFunction(_librapid.pow10)
lerp = _generateFunction(_librapid.lerp)

constPi = _librapid.constPi
constEuler = _librapid.constEuler
constLog2 = _librapid.constLog2
constCatalan = _librapid.constCatalan

norm = _librapid.norm
polar = _librapid.polar

toMpz = _librapid.toMpz
toMpf = _librapid.toMpf
toMpq = _librapid.toMpq
toMpfr = _librapid.toMpfr
