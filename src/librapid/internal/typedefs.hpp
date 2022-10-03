#pragma once

namespace librapid {
	// Forward declarations
	namespace extended {
		struct float16_t;
	}

	using i8   = int8_t;
	using i16  = int16_t;
	using i32  = int32_t;
	using i64  = int64_t;
	using ui8  = uint8_t;
	using ui16 = uint16_t;
	using ui32 = uint32_t;
	using ui64 = uint64_t;
	using f16  = librapid::extended::float16_t;
	using f32  = float;
	using f64  = double;

	using mpz  = mpz_class;
	using mpf  = mpf_class;
	using mpq  = mpq_class;
	using mpfr = mpfr::mpreal;
	// using mpc  = librapid::Complex<mpfr>;

	// Assert all sizes are correct
	static_assert(sizeof(i8) == 1, "i8 is not 1 byte");
	static_assert(sizeof(i16) == 2, "i16 is not 2 bytes");
	static_assert(sizeof(i32) == 4, "i32 is not 4 bytes");
	static_assert(sizeof(i64) == 8, "i64 is not 8 bytes");
	static_assert(sizeof(ui8) == 1, "ui8 is not 1 byte");
	static_assert(sizeof(ui16) == 2, "ui16 is not 2 bytes");
	static_assert(sizeof(ui32) == 4, "ui32 is not 4 bytes");
	static_assert(sizeof(ui64) == 8, "ui64 is not 8 bytes");
	static_assert(sizeof(f32) == 4, "f32 is not 4 bytes");
	static_assert(sizeof(f64) == 8, "f64 is not 8 bytes");
} // namespace librapid
