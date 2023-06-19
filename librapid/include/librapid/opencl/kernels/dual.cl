#ifndef LIBRAPID_OPENCL_DUAL
#define LIBRAPID_OPENCL_DUAL

#define DUAL_DEF(TYPE)                                                                             \
	struct Dual_##TYPE {                                                                           \
		TYPE value;                                                                                \
		TYPE derivative;                                                                           \
	};

DUAL_DEF(int8_t);
DUAL_DEF(int16_t);
DUAL_DEF(int32_t);
DUAL_DEF(int64_t);
DUAL_DEF(uint8_t);
DUAL_DEF(uint16_t);
DUAL_DEF(uint32_t);
DUAL_DEF(uint64_t);
DUAL_DEF(float);
DUAL_DEF(double);

#endif // LIBRAPID_OPENCL_DUAL