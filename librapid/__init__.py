import _librapid


class DataType:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __repr__(self):
        return f"DataType(name={self.name}, size={self.size})"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name and self.size == other.size

    def __hash__(self):
        return hash((self.name, self.size))


class Backend:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Backend(name={self.name})"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Shape(_librapid.Shape):
    """
    Stores the dimensions of an N-dimensional Array.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def Array(*args, **kwargs):
    """
    Creates an N-dimensional Array.

    Parameters
    ----------
    shape : tuple
        The dimensions of the Array.
    dtype : DataType, optional
        The data type of the Array.
    backend : Backend, optional
        The backend of the Array.

    Returns
    -------
    Array
        The created Array.
    """

    # Check for shape/data, dtype, and backend
    shape = None
    data = None
    dtype = None
    backend = None

    for i, arg in enumerate(args):
        if i == 0:
            data = arg
        elif i == 1:
            dtype = arg
        elif i == 2:
            backend = arg

    for key, value in kwargs.items():
        if key == "shape":
            shape = value
        elif key == "data":
            data = value
        elif key == "dtype":
            dtype = value
        elif key == "backend":
            backend = value
        else:
            raise ValueError(f"Unknown argument {key}")

    if shape is None and data is None:
        raise ValueError("Must specify either shape or data")

    if shape is not None and data is not None:
        raise ValueError("Cannot specify both shape and data")

    if dtype is None:
        dtype = float32

    if backend is None:
        backend = CPU

    if backend == CPU:
        arrayType = __cpuTypeList.get(dtype, None)
    elif backend == OpenCL:
        if not _librapid.hasOpenCL():
            raise RuntimeError("OpenCL is not supported in this build "
                               "of librapid. Ensure OpenCL is installed "
                               "on your system and reinstall librapid "
                               "from source.")

        arrayType = __openclTypeList.get(dtype, None)
    elif backend == CUDA:
        if not _librapid.hasCUDA():
            raise RuntimeError("CUDA is not supported in this build "
                               "of librapid. Ensure CUDA is installed "
                               "on your system and reinstall librapid "
                               "from source.")

        arrayType = __cudaTypeList.get(dtype, None)
    else:
        raise ValueError(f"Unknown backend {backend}")

    if arrayType is None:
        raise ValueError(f"Unknown data type {dtype}")

    if shape is not None:
        return arrayType(Shape(shape))
    elif data is not None:
        return arrayType(data)
    else:
        raise RuntimeError("Unknown error")


def hasOpenCL():
    """
    Checks if OpenCL is supported.

    Returns
    -------
    bool
        True if OpenCL is supported, False otherwise.
    """

    return _librapid.hasOpenCL()


def hasCUDA():
    """
    Checks if CUDA is supported.

    Returns
    -------
    bool
        True if CUDA is supported, False otherwise.
    """

    return _librapid.hasCUDA()


def setNumThreads(numThreads):
    """
    Sets the number of threads to use for parallel operations.

    Parameters
    ----------
    numThreads : int
        The number of threads to use.
    """

    _librapid.setNumThreads(numThreads)


def getNumThreads():
    """
    Gets the number of threads used for parallel operations.

    Returns
    -------
    int
        The number of threads used.
    """

    return _librapid.getNumThreads()


def setSeed(seed):
    """
    Sets the seed for random number generation.

    Parameters
    ----------
    seed : int
        The seed to use.
    """

    _librapid.setSeed(seed)


def getSeed():
    """
    Gets the seed for random number generation.

    Returns
    -------
    int
        The seed used.
    """

    return _librapid.getSeed()


transpose = _librapid.transpose
dot = _librapid.dot
sin = _librapid.sin
cos = _librapid.cos
tan = _librapid.tan
asin = _librapid.asin
acos = _librapid.acos
atan = _librapid.atan
sinh = _librapid.sinh
cosh = _librapid.cosh
tanh = _librapid.tanh
asinh = _librapid.asinh
acosh = _librapid.acosh
atanh = _librapid.atanh
sqrt = _librapid.sqrt
cbrt = _librapid.cbrt
log = _librapid.log
log2 = _librapid.log2
log10 = _librapid.log10
exp = _librapid.exp
exp2 = _librapid.exp2
exp10 = _librapid.exp10

float32 = DataType("float32", 4)
float64 = DataType("float64", 8)
int32 = DataType("int32", 4)
int64 = DataType("int64", 8)
uint32 = DataType("uint32", 4)
uint64 = DataType("uint64", 8)
# Complex32 = DataType("Complex32", 8)
# Complex64 = DataType("Complex64", 16)

__cpuTypeList = {
    int32: _librapid.ArrayInt32CPU,
    int64: _librapid.ArrayInt64CPU,
    uint32: _librapid.ArrayUInt32CPU,
    uint64: _librapid.ArrayUInt64CPU,
    float32: _librapid.ArrayFloatCPU,
    float64: _librapid.ArrayDoubleCPU,
    # Complex32: _librapid.ArrayComplexFloatCPU,
    # Complex64: _librapid.ArrayComplexDoubleCPU,
}

if _librapid.hasOpenCL():
    __openclTypeList = {
        int32: _librapid.ArrayInt32OpenCL,
        int64: _librapid.ArrayInt64OpenCL,
        uint32: _librapid.ArrayUInt32OpenCL,
        uint64: _librapid.ArrayUInt64OpenCL,
        float32: _librapid.ArrayFloatOpenCL,
        float64: _librapid.ArrayDoubleOpenCL,
        # Complex32: _librapid.ArrayComplexFloatOpenCL,
        # Complex64: _librapid.ArrayComplexDoubleOpenCL,
    }
else:
    __openclTypeList = None

if _librapid.hasCUDA():
    __cudaTypeList = {
        int32: _librapid.ArrayInt32CUDA,
        int64: _librapid.ArrayInt64CUDA,
        uint32: _librapid.ArrayUInt32CUDA,
        uint64: _librapid.ArrayUInt64CUDA,
        float32: _librapid.ArrayFloatCUDA,
        float64: _librapid.ArrayDoubleCUDA,
        # Complex32: _librapid.ArrayComplexFloatCUDA,
        # Complex64: _librapid.ArrayComplexDoubleCUDA,
    }
else:
    __cudaTypeList = None

CPU = Backend("CPU")
OpenCL = Backend("OpenCL")
CUDA = Backend("CUDA")
