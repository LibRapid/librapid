# NOTE: Here, None represents a default, unintialized value

typeMap = {
	"ArrayB" : ("b", "bool", "boolean"),
	"ArrayC" : ("c", "char", "byte", "character"),
	"ArrayF16" : ("half", "float16", "f16"),
	"ArrayF32" : ("float", "float32", "f32", "default", None),
	"ArrayF64" : ("double", "float64", "f64"),
	"ArrayI16" : ("short", "int16", "i16"),
	"ArrayI32" : ("int", "long", "int32", "i32"),
	"ArrayI64" : ("long long", "int64", "i64"),
	"ArrayMPZ" : ("bigint", "mpz", "mpz_t", "mpz_class", "mpir_int"),
	"ArrayMPQ" : ("bigrational", "mpq", "mpq_t", "mpq_class", "mpir_rational"),
	"ArrayMPFR" : ("bigfloat", "mpfr", "mpreal", "mpfr_float"),
	"ArrayCF32" : ("complex float", "cfloat", "cf32", "c32"),
	"ArrayCF64" : ("compled double", "cdouble", "cf64", "c64"),
	"ArrayCMPFR" : ("complex multiprecision", "complex multiprec", "cmpfr", "cmpf", "cmp")
}

deviceMap = {
	"CPU" : ("cpu", "host", "default", None),
	"GPU" : ("gpu", "cuda", "device")
}

def mapType(type:str = None, device:str = None):
	resType = None
	resDevice = None
	
	for key, val in typeMap.items():
		if type in val:
			resType = key

	for key, val in deviceMap.items():
		if device in val:
			resDevice = key

	return resType, resDevice
