#ifndef LIBRAPID_MATH_HALF_HPP
#define LIBRAPID_MATH_HALF_HPP

//
// inspired by:
// https://github.com/acgessler/half_float
// https://github.com/x448/float16
//

namespace librapid {
	namespace detail {
		union float16_t {
			uint16_t m_bits;
			struct {
				uint16_t m_frac : 10;
				uint16_t m_exp : 5;
				uint16_t m_sign : 1;
			} m_ieee;
		};

		union float32_t {
			uint32_t m_bits;
			struct {
				uint32_t m_frac : 23;
				uint32_t m_exp : 8;
				uint32_t m_sign : 1;
			} m_ieee;
			float m_float;
		};

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint32_t
		uint32Sels(uint32_t test, uint32_t a, uint32_t b) noexcept {
			const uint32_t mask	  = (((std::int32_t)test) >> 31);
			const uint32_t sel_a  = (a & mask);
			const uint32_t sel_b  = (b & ~mask);
			const uint32_t result = (sel_a | sel_b);
			return (result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint32_t
		uint32Selb(uint32_t mask, uint32_t a, uint32_t b) noexcept {
			const uint32_t sel_a  = (a & mask);
			const uint32_t sel_b  = (b & ~mask);
			const uint32_t result = (sel_a | sel_b);
			return (result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint16_t
		uint16Sels(uint16_t test, uint16_t a, uint16_t b) noexcept {
			const uint16_t mask	  = (((int16_t)test) >> 15);
			const uint16_t sel_a  = (a & mask);
			const uint16_t sel_b  = (b & ~mask);
			const uint16_t result = (sel_a | sel_b);
			return (result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint32_t
		uint32Cntlz(uint32_t x) noexcept {
#if defined(LIBRAPID_GNU_CXX)
			uint32_t is_x_nez_msb = (-x);
			uint32_t nlz		  = __builtin_clz(x);
			uint32_t result		  = _uint32_sels(is_x_nez_msb, nlz, 0x00000020);
			return (result);
#else
			const uint32_t x0  = (x >> 1);
			const uint32_t x1  = (x | x0);
			const uint32_t x2  = (x1 >> 2);
			const uint32_t x3  = (x1 | x2);
			const uint32_t x4  = (x3 >> 4);
			const uint32_t x5  = (x3 | x4);
			const uint32_t x6  = (x5 >> 8);
			const uint32_t x7  = (x5 | x6);
			const uint32_t x8  = (x7 >> 16);
			const uint32_t x9  = (x7 | x8);
			const uint32_t xA  = (~x9);
			const uint32_t xB  = (xA >> 1);
			const uint32_t xC  = (xB & 0x55555555);
			const uint32_t xD  = (xA - xC);
			const uint32_t xE  = (xD & 0x33333333);
			const uint32_t xF  = (xD >> 2);
			const uint32_t x10 = (xF & 0x33333333);
			const uint32_t x11 = (xE + x10);
			const uint32_t x12 = (x11 >> 4);
			const uint32_t x13 = (x11 + x12);
			const uint32_t x14 = (x13 & 0x0f0f0f0f);
			const uint32_t x15 = (x14 >> 8);
			const uint32_t x16 = (x14 + x15);
			const uint32_t x17 = (x16 >> 16);
			const uint32_t x18 = (x16 + x17);
			const uint32_t x19 = (x18 & 0x0000003f);
			return (x19);
#endif
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint16_t
		uint16Cntlz(uint16_t x) noexcept {
#if defined(LIBRAPID_GNU_CXX)
			uint16_t nlz32 = (uint16_t)_uint32_cntlz((uint32_t)x);
			uint32_t nlz   = (nlz32 - 16);
			return (nlz);
#else
			const uint16_t x0  = (x >> 1);
			const uint16_t x1  = (x | x0);
			const uint16_t x2  = (x1 >> 2);
			const uint16_t x3  = (x1 | x2);
			const uint16_t x4  = (x3 >> 4);
			const uint16_t x5  = (x3 | x4);
			const uint16_t x6  = (x5 >> 8);
			const uint16_t x7  = (x5 | x6);
			const uint16_t x8  = (~x7);
			const uint16_t x9  = ((x8 >> 1) & 0x5555);
			const uint16_t xA  = (x8 - x9);
			const uint16_t xB  = (xA & 0x3333);
			const uint16_t xC  = ((xA >> 2) & 0x3333);
			const uint16_t xD  = (xB + xC);
			const uint16_t xE  = (xD >> 4);
			const uint16_t xF  = ((xD + xE) & 0x0f0f);
			const uint16_t x10 = (xF >> 8);
			const uint16_t x11 = ((xF + x10) & 0x001f);
			return (x11);
#endif
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint16_t
		floatToHalf(uint32_t f) noexcept {
			const uint32_t one						= (0x00000001);
			const uint32_t f_s_mask					= (0x80000000);
			const uint32_t f_e_mask					= (0x7f800000);
			const uint32_t f_m_mask					= (0x007fffff);
			const uint32_t f_m_hidden_bit			= (0x00800000);
			const uint32_t f_m_round_bit			= (0x00001000);
			const uint32_t f_snan_mask				= (0x7fc00000);
			const uint32_t f_e_pos					= (0x00000017);
			const uint32_t h_e_pos					= (0x0000000a);
			const uint32_t h_e_mask					= (0x00007c00);
			const uint32_t h_snan_mask				= (0x00007e00);
			const uint32_t h_e_mask_value			= (0x0000001f);
			const uint32_t f_h_s_pos_offset			= (0x00000010);
			const uint32_t f_h_bias_offset			= (0x00000070);
			const uint32_t f_h_m_pos_offset			= (0x0000000d);
			const uint32_t h_nan_min				= (0x00007c01);
			const uint32_t f_h_e_biased_flag		= (0x0000008f);
			const uint32_t f_s						= (f & f_s_mask);
			const uint32_t f_e						= (f & f_e_mask);
			const uint16_t h_s						= (f_s >> f_h_s_pos_offset);
			const uint32_t f_m						= (f & f_m_mask);
			const uint16_t f_e_amount				= (f_e >> f_e_pos);
			const uint32_t f_e_half_bias			= (f_e_amount - f_h_bias_offset);
			const uint32_t f_snan					= (f & f_snan_mask);
			const uint32_t f_m_round_mask			= (f_m & f_m_round_bit);
			const uint32_t f_m_round_offset			= (f_m_round_mask << one);
			const uint32_t f_m_rounded				= (f_m + f_m_round_offset);
			const uint32_t f_m_denorm_sa			= (one - f_e_half_bias);
			const uint32_t f_m_with_hidden			= (f_m_rounded | f_m_hidden_bit);
			const uint32_t f_m_denorm				= (f_m_with_hidden >> f_m_denorm_sa);
			const uint32_t h_m_denorm				= (f_m_denorm >> f_h_m_pos_offset);
			const uint32_t f_m_rounded_overflow		= (f_m_rounded & f_m_hidden_bit);
			const uint32_t m_nan					= (f_m >> f_h_m_pos_offset);
			const uint32_t h_em_nan					= (h_e_mask | m_nan);
			const uint32_t h_e_norm_overflow_offset = (f_e_half_bias + 1);
			const uint32_t h_e_norm_overflow		= (h_e_norm_overflow_offset << h_e_pos);
			const uint32_t h_e_norm					= (f_e_half_bias << h_e_pos);
			const uint32_t h_m_norm					= (f_m_rounded >> f_h_m_pos_offset);
			const uint32_t h_em_norm				= (h_e_norm | h_m_norm);
			const uint32_t is_h_ndenorm_msb			= (f_h_bias_offset - f_e_amount);
			const uint32_t is_f_e_flagged_msb		= (f_h_e_biased_flag - f_e_half_bias);
			const uint32_t is_h_denorm_msb			= (~is_h_ndenorm_msb);
			const uint32_t is_f_m_eqz_msb			= (f_m - 1);
			const uint32_t is_h_nan_eqz_msb			= (m_nan - 1);
			const uint32_t is_f_inf_msb				= (is_f_e_flagged_msb & is_f_m_eqz_msb);
			const uint32_t is_f_nan_underflow_msb	= (is_f_e_flagged_msb & is_h_nan_eqz_msb);
			const uint32_t is_e_overflow_msb		= (h_e_mask_value - f_e_half_bias);
			const uint32_t is_h_inf_msb				= (is_e_overflow_msb | is_f_inf_msb);
			const uint32_t is_f_nsnan_msb			= (f_snan - f_snan_mask);
			const uint32_t is_m_norm_overflow_msb	= (-f_m_rounded_overflow);
			const uint32_t is_f_snan_msb			= (~is_f_nsnan_msb);
			const uint32_t h_em_overflow_result =
			  uint32Sels(is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm);
			const uint32_t h_em_nan_result =
			  uint32Sels(is_f_e_flagged_msb, h_em_nan, h_em_overflow_result);
			const uint32_t h_em_nan_underflow_result =
			  uint32Sels(is_f_nan_underflow_msb, h_nan_min, h_em_nan_result);
			const uint32_t h_em_inf_result =
			  uint32Sels(is_h_inf_msb, h_e_mask, h_em_nan_underflow_result);
			const uint32_t h_em_denorm_result =
			  uint32Sels(is_h_denorm_msb, h_m_denorm, h_em_inf_result);
			const uint32_t h_em_snan_result =
			  uint32Sels(is_f_snan_msb, h_snan_mask, h_em_denorm_result);
			const uint32_t h_result = (h_s | h_em_snan_result);
			return (uint16_t)(h_result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint32_t
		halfToFloat(uint16_t h) noexcept {
			const uint32_t h_e_mask				= (0x00007c00);
			const uint32_t h_m_mask				= (0x000003ff);
			const uint32_t h_s_mask				= (0x00008000);
			const uint32_t h_f_s_pos_offset		= (0x00000010);
			const uint32_t h_f_e_pos_offset		= (0x0000000d);
			const uint32_t h_f_bias_offset		= (0x0001c000);
			const uint32_t f_e_mask				= (0x7f800000);
			const uint32_t f_m_mask				= (0x007fffff);
			const uint32_t h_f_e_denorm_bias	= (0x0000007e);
			const uint32_t h_f_m_denorm_sa_bias = (0x00000008);
			const uint32_t f_e_pos				= (0x00000017);
			const uint32_t h_e_mask_minus_one	= (0x00007bff);
			const uint32_t h_e					= (h & h_e_mask);
			const uint32_t h_m					= (h & h_m_mask);
			const uint32_t h_s					= (h & h_s_mask);
			const uint32_t h_e_f_bias			= (h_e + h_f_bias_offset);
			const uint32_t h_m_nlz				= uint32Cntlz(h_m);
			const uint32_t f_s					= (h_s << h_f_s_pos_offset);
			const uint32_t f_e					= (h_e_f_bias << h_f_e_pos_offset);
			const uint32_t f_m					= (h_m << h_f_e_pos_offset);
			const uint32_t f_em					= (f_e | f_m);
			const uint32_t h_f_m_sa				= (h_m_nlz - h_f_m_denorm_sa_bias);
			const uint32_t f_e_denorm_unpacked	= (h_f_e_denorm_bias - h_f_m_sa);
			const uint32_t h_f_m				= (h_m << h_f_m_sa);
			const uint32_t f_m_denorm			= (h_f_m & f_m_mask);
			const uint32_t f_e_denorm			= (f_e_denorm_unpacked << f_e_pos);
			const uint32_t f_em_denorm			= (f_e_denorm | f_m_denorm);
			const uint32_t f_em_nan				= (f_e_mask | f_m);
			const uint32_t is_e_eqz_msb			= (h_e - 1);
			const uint32_t is_m_nez_msb			= (-h_m);
			const uint32_t is_e_flagged_msb		= (h_e_mask_minus_one - h_e);
			const uint32_t is_zero_msb			= (is_e_eqz_msb & ~is_m_nez_msb);
			const uint32_t is_inf_msb			= (is_e_flagged_msb & ~is_m_nez_msb);
			const uint32_t is_denorm_msb		= (is_m_nez_msb & is_e_eqz_msb);
			const uint32_t is_nan_msb			= (is_e_flagged_msb & is_m_nez_msb);
			const uint32_t is_zero				= (((std::int32_t)is_zero_msb) >> 31);
			const uint32_t f_zero_result		= (f_em & ~is_zero);
			const uint32_t f_denorm_result = uint32Sels(is_denorm_msb, f_em_denorm, f_zero_result);
			const uint32_t f_inf_result	   = uint32Sels(is_inf_msb, f_e_mask, f_denorm_result);
			const uint32_t f_nan_result	   = uint32Sels(is_nan_msb, f_em_nan, f_inf_result);
			const uint32_t f_result		   = (f_s | f_nan_result);
			return (f_result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint16_t halfAdd(uint16_t x,
																			 uint16_t y) noexcept {
			constexpr uint16_t one				  = (0x0001);
			constexpr uint16_t msb_to_lsb_sa	  = (0x000f);
			constexpr uint16_t h_s_mask			  = (0x8000);
			constexpr uint16_t h_e_mask			  = (0x7c00);
			constexpr uint16_t h_m_mask			  = (0x03ff);
			constexpr uint16_t h_m_msb_mask		  = (0x2000);
			constexpr uint16_t h_m_msb_sa		  = (0x000d);
			constexpr uint16_t h_m_hidden		  = (0x0400);
			constexpr uint16_t h_e_pos			  = (0x000a);
			constexpr uint16_t h_e_bias_minus_one = (0x000e);
			constexpr uint16_t h_m_grs_carry	  = (0x4000);
			constexpr uint16_t h_m_grs_carry_pos  = (0x000e);
			constexpr uint16_t h_grs_size		  = (0x0003);
			constexpr uint16_t h_snan			  = (0xfe00);
			constexpr uint16_t h_e_mask_minus_one = (0x7bff);
			const uint16_t h_grs_round_carry	  = (one << h_grs_size);
			const uint16_t h_grs_round_mask		  = (h_grs_round_carry - one);
			const uint16_t x_e					  = (x & h_e_mask);
			const uint16_t y_e					  = (y & h_e_mask);
			const uint16_t is_y_e_larger_msb	  = (x_e - y_e);
			const uint16_t a					  = uint16Sels(is_y_e_larger_msb, y, x);
			const uint16_t a_s					  = (a & h_s_mask);
			const uint16_t a_e					  = (a & h_e_mask);
			const uint16_t a_m_no_hidden_bit	  = (a & h_m_mask);
			const uint16_t a_em_no_hidden_bit	  = (a_e | a_m_no_hidden_bit);
			const uint16_t b					  = uint16Sels(is_y_e_larger_msb, x, y);
			const uint16_t b_s					  = (b & h_s_mask);
			const uint16_t b_e					  = (b & h_e_mask);
			const uint16_t b_m_no_hidden_bit	  = (b & h_m_mask);
			const uint16_t b_em_no_hidden_bit	  = (b_e | b_m_no_hidden_bit);
			const uint16_t is_diff_sign_msb		  = (a_s ^ b_s);
			const uint16_t is_a_inf_msb			  = (h_e_mask_minus_one - a_em_no_hidden_bit);
			const uint16_t is_b_inf_msb			  = (h_e_mask_minus_one - b_em_no_hidden_bit);
			const uint16_t is_undenorm_msb		  = (a_e - 1);
			const uint16_t is_undenorm			  = (((int16_t)is_undenorm_msb) >> 15);
			const uint16_t is_both_inf_msb		  = (is_a_inf_msb & is_b_inf_msb);
			const uint16_t is_invalid_inf_op_msb  = (is_both_inf_msb & b_s);
			const uint16_t is_a_e_nez_msb		  = (-a_e);
			const uint16_t is_b_e_nez_msb		  = (-b_e);
			const uint16_t is_a_e_nez			  = (((int16_t)is_a_e_nez_msb) >> 15);
			const uint16_t is_b_e_nez			  = (((int16_t)is_b_e_nez_msb) >> 15);
			const uint16_t a_m_hidden_bit		  = (is_a_e_nez & h_m_hidden);
			const uint16_t b_m_hidden_bit		  = (is_b_e_nez & h_m_hidden);
			const uint16_t a_m_no_grs			  = (a_m_no_hidden_bit | a_m_hidden_bit);
			const uint16_t b_m_no_grs			  = (b_m_no_hidden_bit | b_m_hidden_bit);
			const uint16_t diff_e				  = (a_e - b_e);
			const uint16_t a_e_unbias			  = (a_e - h_e_bias_minus_one);
			const uint16_t a_m					  = (a_m_no_grs << h_grs_size);
			const uint16_t a_e_biased			  = (a_e >> h_e_pos);
			const uint16_t m_sa_unbias			  = (a_e_unbias >> h_e_pos);
			const uint16_t m_sa_default			  = (diff_e >> h_e_pos);
			const uint16_t m_sa_unbias_mask		  = (is_a_e_nez_msb & ~is_b_e_nez_msb);
			const uint16_t m_sa			 = uint16Sels(m_sa_unbias_mask, m_sa_unbias, m_sa_default);
			const uint16_t b_m_no_sticky = (b_m_no_grs << h_grs_size);
			const uint16_t sh_m			 = (b_m_no_sticky >> m_sa);
			const uint16_t sticky_overflow	 = (one << m_sa);
			const uint16_t sticky_mask		 = (sticky_overflow - 1);
			const uint16_t sticky_collect	 = (b_m_no_sticky & sticky_mask);
			const uint16_t is_sticky_set_msb = (-sticky_collect);
			const uint16_t sticky			 = (is_sticky_set_msb >> msb_to_lsb_sa);
			const uint16_t b_m				 = (sh_m | sticky);
			const uint16_t is_c_m_ab_pos_msb = (b_m - a_m);
			const uint16_t c_inf			 = (a_s | h_e_mask);
			const uint16_t c_m_sum			 = (a_m + b_m);
			const uint16_t c_m_diff_ab		 = (a_m - b_m);
			const uint16_t c_m_diff_ba		 = (b_m - a_m);
			const uint16_t c_m_smag_diff = uint16Sels(is_c_m_ab_pos_msb, c_m_diff_ab, c_m_diff_ba);
			const uint16_t c_s_diff		 = uint16Sels(is_c_m_ab_pos_msb, a_s, b_s);
			const uint16_t c_s			 = uint16Sels(is_diff_sign_msb, c_s_diff, a_s);
			const uint16_t c_m_smag_diff_nlz  = uint16Cntlz(c_m_smag_diff);
			const uint16_t diff_norm_sa		  = (c_m_smag_diff_nlz - one);
			const uint16_t is_diff_denorm_msb = (a_e_biased - diff_norm_sa);
			const uint16_t is_diff_denorm	  = (((int16_t)is_diff_denorm_msb) >> 15);
			const uint16_t is_a_or_b_norm_msb = (-a_e_biased);
			const uint16_t diff_denorm_sa	  = (a_e_biased - 1);
			const uint16_t c_m_diff_denorm	  = (c_m_smag_diff << diff_denorm_sa);
			const uint16_t c_m_diff_norm	  = (c_m_smag_diff << diff_norm_sa);
			const uint16_t c_e_diff_norm	  = (a_e_biased - diff_norm_sa);
			const uint16_t c_m_diff_ab_norm =
			  uint16Sels(is_diff_denorm_msb, c_m_diff_denorm, c_m_diff_norm);
			const uint16_t c_e_diff_ab_norm = (c_e_diff_norm & ~is_diff_denorm);
			const uint16_t c_m_diff =
			  uint16Sels(is_a_or_b_norm_msb, c_m_diff_ab_norm, c_m_smag_diff);
			const uint16_t c_e_diff = uint16Sels(is_a_or_b_norm_msb, c_e_diff_ab_norm, a_e_biased);
			const uint16_t is_diff_eqz_msb			= (c_m_diff - 1);
			const uint16_t is_diff_exactly_zero_msb = (is_diff_sign_msb & is_diff_eqz_msb);
			const uint16_t is_diff_exactly_zero		= (((int16_t)is_diff_exactly_zero_msb) >> 15);
			const uint16_t c_m_added		 = uint16Sels(is_diff_sign_msb, c_m_diff, c_m_sum);
			const uint16_t c_e_added		 = uint16Sels(is_diff_sign_msb, c_e_diff, a_e_biased);
			const uint16_t c_m_carry		 = (c_m_added & h_m_grs_carry);
			const uint16_t is_c_m_carry_msb	 = (-c_m_carry);
			const uint16_t c_e_hidden_offset = ((c_m_added & h_m_grs_carry) >> h_m_grs_carry_pos);
			const uint16_t c_m_sub_hidden	 = (c_m_added >> one);
			const uint16_t c_m_no_hidden = uint16Sels(is_c_m_carry_msb, c_m_sub_hidden, c_m_added);
			const uint16_t c_e_no_hidden = (c_e_added + c_e_hidden_offset);
			const uint16_t c_m_no_hidden_msb  = (c_m_no_hidden & h_m_msb_mask);
			const uint16_t undenorm_m_msb_odd = (c_m_no_hidden_msb >> h_m_msb_sa);
			const uint16_t undenorm_fix_e	  = (is_undenorm & undenorm_m_msb_odd);
			const uint16_t c_e_fixed		  = (c_e_no_hidden + undenorm_fix_e);
			const uint16_t c_m_round_amount	  = (c_m_no_hidden & h_grs_round_mask);
			const uint16_t c_m_rounded		  = (c_m_no_hidden + c_m_round_amount);
			const uint16_t c_m_round_overflow =
			  ((c_m_rounded & h_m_grs_carry) >> h_m_grs_carry_pos);
			const uint16_t c_e_rounded	 = (c_e_fixed + c_m_round_overflow);
			const uint16_t c_m_no_grs	 = ((c_m_rounded >> h_grs_size) & h_m_mask);
			const uint16_t c_e			 = (c_e_rounded << h_e_pos);
			const uint16_t c_em			 = (c_e | c_m_no_grs);
			const uint16_t c_normal		 = (c_s | c_em);
			const uint16_t c_inf_result	 = uint16Sels(is_a_inf_msb, c_inf, c_normal);
			const uint16_t c_zero_result = (c_inf_result & ~is_diff_exactly_zero);
			const uint16_t c_result		 = uint16Sels(is_invalid_inf_op_msb, h_snan, c_zero_result);
			return (c_result);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr uint16_t halfMul(uint16_t x,
																			 uint16_t y) noexcept {
			const uint32_t one					   = (0x00000001);
			const uint32_t h_s_mask				   = (0x00008000);
			const uint32_t h_e_mask				   = (0x00007c00);
			const uint32_t h_m_mask				   = (0x000003ff);
			const uint32_t h_m_hidden			   = (0x00000400);
			const uint32_t h_e_pos				   = (0x0000000a);
			const uint32_t h_e_bias				   = (0x0000000f);
			const uint32_t h_m_bit_count		   = (0x0000000a);
			const uint32_t h_m_bit_half_count	   = (0x00000005);
			const uint32_t h_nan_min			   = (0x00007c01);
			const uint32_t h_e_mask_minus_one	   = (0x00007bff);
			const uint32_t h_snan				   = (0x0000fe00);
			const uint32_t m_round_overflow_bit	   = (0x00000020);
			const uint32_t m_hidden_bit			   = (0x00100000);
			const uint32_t a_s					   = (x & h_s_mask);
			const uint32_t b_s					   = (y & h_s_mask);
			const uint32_t c_s					   = (a_s ^ b_s);
			const uint32_t x_e					   = (x & h_e_mask);
			const uint32_t x_e_eqz_msb			   = (x_e - 1);
			const uint32_t a					   = uint32Sels(x_e_eqz_msb, y, x);
			const uint32_t b					   = uint32Sels(x_e_eqz_msb, x, y);
			const uint32_t a_e					   = (a & h_e_mask);
			const uint32_t b_e					   = (b & h_e_mask);
			const uint32_t a_m					   = (a & h_m_mask);
			const uint32_t b_m					   = (b & h_m_mask);
			const uint32_t a_e_amount			   = (a_e >> h_e_pos);
			const uint32_t b_e_amount			   = (b_e >> h_e_pos);
			const uint32_t a_m_with_hidden		   = (a_m | h_m_hidden);
			const uint32_t b_m_with_hidden		   = (b_m | h_m_hidden);
			const uint32_t c_m_normal			   = (a_m_with_hidden * b_m_with_hidden);
			const uint32_t c_m_denorm_biased	   = (a_m_with_hidden * b_m);
			const uint32_t c_e_denorm_unbias_e	   = (h_e_bias - a_e_amount);
			const uint32_t c_m_denorm_round_amount = (c_m_denorm_biased & h_m_mask);
			const uint32_t c_m_denorm_rounded	   = (c_m_denorm_biased + c_m_denorm_round_amount);
			const uint32_t c_m_denorm_inplace	   = (c_m_denorm_rounded >> h_m_bit_count);
			const uint32_t c_m_denorm_unbiased	   = (c_m_denorm_inplace >> c_e_denorm_unbias_e);
			const uint32_t c_m_denorm			   = (c_m_denorm_unbiased & h_m_mask);
			const uint32_t c_e_amount_biased	   = (a_e_amount + b_e_amount);
			const uint32_t c_e_amount_unbiased	   = (c_e_amount_biased - h_e_bias);
			const uint32_t is_c_e_unbiased_underflow = (((std::int32_t)c_e_amount_unbiased) >> 31);
			const uint32_t c_e_underflow_half_sa	 = (-c_e_amount_unbiased);
			const uint32_t c_e_underflow_sa			 = (c_e_underflow_half_sa << one);
			const uint32_t c_m_underflow			 = (c_m_normal >> c_e_underflow_sa);
			const uint32_t c_e_underflow_added = (c_e_amount_unbiased & ~is_c_e_unbiased_underflow);
			const uint32_t c_m_underflow_added =
			  uint32Selb(is_c_e_unbiased_underflow, c_m_underflow, c_m_normal);
			const uint32_t is_mul_overflow_test		 = (c_e_underflow_added & m_round_overflow_bit);
			const uint32_t is_mul_overflow_msb		 = (-is_mul_overflow_test);
			const uint32_t c_e_norm_radix_corrected	 = (c_e_underflow_added + 1);
			const uint32_t c_m_norm_radix_corrected	 = (c_m_underflow_added >> one);
			const uint32_t c_m_norm_hidden_bit		 = (c_m_norm_radix_corrected & m_hidden_bit);
			const uint32_t is_c_m_norm_no_hidden_msb = (c_m_norm_hidden_bit - 1);
			const uint32_t c_m_norm_lo	   = (c_m_norm_radix_corrected >> h_m_bit_half_count);
			const uint32_t c_m_norm_lo_nlz = uint16Cntlz(c_m_norm_lo);
			const uint32_t is_c_m_hidden_nunderflow_msb =
			  (c_m_norm_lo_nlz - c_e_norm_radix_corrected);
			const uint32_t is_c_m_hidden_underflow_msb = (~is_c_m_hidden_nunderflow_msb);
			const uint32_t is_c_m_hidden_underflow =
			  (((std::int32_t)is_c_m_hidden_underflow_msb) >> 31);
			const uint32_t c_m_hidden_underflow_normalized_sa = (c_m_norm_lo_nlz >> one);
			const uint32_t c_m_hidden_underflow_normalized =
			  (c_m_norm_radix_corrected << c_m_hidden_underflow_normalized_sa);
			const uint32_t c_m_hidden_normalized = (c_m_norm_radix_corrected << c_m_norm_lo_nlz);
			const uint32_t c_e_hidden_normalized = (c_e_norm_radix_corrected - c_m_norm_lo_nlz);
			const uint32_t c_e_hidden = (c_e_hidden_normalized & ~is_c_m_hidden_underflow);
			const uint32_t c_m_hidden = uint32Sels(
			  is_c_m_hidden_underflow_msb, c_m_hidden_underflow_normalized, c_m_hidden_normalized);
			const uint32_t c_m_normalized =
			  uint32Sels(is_c_m_norm_no_hidden_msb, c_m_hidden, c_m_norm_radix_corrected);
			const uint32_t c_e_normalized =
			  uint32Sels(is_c_m_norm_no_hidden_msb, c_e_hidden, c_e_norm_radix_corrected);
			const uint32_t c_m_norm_round_amount  = (c_m_normalized & h_m_mask);
			const uint32_t c_m_norm_rounded		  = (c_m_normalized + c_m_norm_round_amount);
			const uint32_t is_round_overflow_test = (c_e_normalized & m_round_overflow_bit);
			const uint32_t is_round_overflow_msb  = (-is_round_overflow_test);
			const uint32_t c_m_norm_inplace		  = (c_m_norm_rounded >> h_m_bit_count);
			const uint32_t c_m					  = (c_m_norm_inplace & h_m_mask);
			const uint32_t c_e_norm_inplace		  = (c_e_normalized << h_e_pos);
			const uint32_t c_e					  = (c_e_norm_inplace & h_e_mask);
			const uint32_t c_em_nan				  = (h_e_mask | a_m);
			const uint32_t c_nan				  = (a_s | c_em_nan);
			const uint32_t c_denorm				  = (c_s | c_m_denorm);
			const uint32_t c_inf				  = (c_s | h_e_mask);
			const uint32_t c_em_norm			  = (c_e | c_m);
			const uint32_t is_a_e_flagged_msb	  = (h_e_mask_minus_one - a_e);
			const uint32_t is_b_e_flagged_msb	  = (h_e_mask_minus_one - b_e);
			const uint32_t is_a_e_eqz_msb		  = (a_e - 1);
			const uint32_t is_a_m_eqz_msb		  = (a_m - 1);
			const uint32_t is_b_e_eqz_msb		  = (b_e - 1);
			const uint32_t is_b_m_eqz_msb		  = (b_m - 1);
			const uint32_t is_b_eqz_msb			  = (is_b_e_eqz_msb & is_b_m_eqz_msb);
			const uint32_t is_a_eqz_msb			  = (is_a_e_eqz_msb & is_a_m_eqz_msb);
			const uint32_t is_c_nan_via_a_msb	  = (is_a_e_flagged_msb & ~is_b_e_flagged_msb);
			const uint32_t is_c_nan_via_b_msb	  = (is_b_e_flagged_msb & ~is_b_m_eqz_msb);
			const uint32_t is_c_nan_msb			  = (is_c_nan_via_a_msb | is_c_nan_via_b_msb);
			const uint32_t is_c_denorm_msb		  = (is_b_e_eqz_msb & ~is_a_e_flagged_msb);
			const uint32_t is_a_inf_msb			  = (is_a_e_flagged_msb & is_a_m_eqz_msb);
			const uint32_t is_c_snan_msb		  = (is_a_inf_msb & is_b_eqz_msb);
			const uint32_t is_c_nan_min_via_a_msb = (is_a_e_flagged_msb & is_b_eqz_msb);
			const uint32_t is_c_nan_min_via_b_msb = (is_b_e_flagged_msb & is_a_eqz_msb);
			const uint32_t is_c_nan_min_msb		= (is_c_nan_min_via_a_msb | is_c_nan_min_via_b_msb);
			const uint32_t is_c_inf_msb			= (is_a_e_flagged_msb | is_b_e_flagged_msb);
			const uint32_t is_overflow_msb		= (is_round_overflow_msb | is_mul_overflow_msb);
			const uint32_t c_em_overflow_result = uint32Sels(is_overflow_msb, h_e_mask, c_em_norm);
			const uint32_t c_common_result		= (c_s | c_em_overflow_result);
			const uint32_t c_zero_result		= uint32Sels(is_b_eqz_msb, c_s, c_common_result);
			const uint32_t c_nan_result			= uint32Sels(is_c_nan_msb, c_nan, c_zero_result);
			const uint32_t c_nan_min_result = uint32Sels(is_c_nan_min_msb, h_nan_min, c_nan_result);
			const uint32_t c_inf_result		= uint32Sels(is_c_inf_msb, c_inf, c_nan_min_result);
			const uint32_t c_denorm_result	= uint32Sels(is_c_denorm_msb, c_denorm, c_inf_result);
			const uint32_t c_result			= uint32Sels(is_c_snan_msb, h_snan, c_denorm_result);
			return (uint16_t)(c_result);
		}

		constexpr inline uint16_t halfNeg(uint16_t h) noexcept { return h ^ 0x8000; }

		constexpr inline uint16_t halfSub(uint16_t ha, uint16_t hb) noexcept {
			return halfAdd(ha, halfNeg(hb));
		}
	} // namespace detail

	class half {
	public:
		half() noexcept	   = default;
		half(const half &) = default;
		half(half &&)	   = default;

		LIBRAPID_ALWAYS_INLINE half(float f) noexcept;

		template<typename T>
		LIBRAPID_ALWAYS_INLINE explicit half(T d) noexcept;

		half &operator=(const half &) = default;
		half &operator=(half &&)	  = default;

		template<typename T>
		LIBRAPID_ALWAYS_INLINE half &operator=(T d) noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static half fromBits(uint16_t bits) noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator float() const noexcept;

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator T() const noexcept;

		LIBRAPID_ALWAYS_INLINE half &operator+=(const half &rhs) noexcept;
		LIBRAPID_ALWAYS_INLINE half &operator-=(const half &rhs) noexcept;
		LIBRAPID_ALWAYS_INLINE half &operator*=(const half &rhs) noexcept;
		LIBRAPID_ALWAYS_INLINE half &operator/=(const half &rhs) noexcept;

		LIBRAPID_ALWAYS_INLINE half &operator--() noexcept;
		LIBRAPID_ALWAYS_INLINE half operator--(int) noexcept;
		LIBRAPID_ALWAYS_INLINE half &operator++() noexcept;
		LIBRAPID_ALWAYS_INLINE half operator++(int) noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator-() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator+() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::float16_t data() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::float16_t &data() noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE std::string
		str(const std::string &format = "{}") const;

		//		static half infinity;
		//		static half max;
		//		static half maxSubnormal;
		//		static half min;
		//		static half minPositive;
		//		static half minPositiveSubnormal;
		//		static half nan;
		//		static half negativeInfinity;
		//		static half epsilon;
		//
		//		static half one;
		//		static half negativeOne;
		//		static half two;
		//		static half negativeTwo;
		//		static half half_;
		//		static half negativeHalf;
		//		static half zero;
		//		static half negativeZero;
		//		static half e;
		//		static half pi;

	private:
		detail::float16_t m_value;
	};

	half::half(float f) noexcept {
		detail::float32_t tmp;
		tmp.m_float	   = f;
		m_value.m_bits = detail::floatToHalf(tmp.m_bits);
	}

	template<typename T>
	half::half(T d) noexcept : half(static_cast<float>(d)) {}

	template<typename T>
	half &half::operator=(T d) noexcept {
		*this = half(d);
		return *this;
	}

	half half::fromBits(uint16_t bits) noexcept {
		half h;
		h.m_value.m_bits = bits;
		return h;
	}

	half::operator float() const noexcept {
		detail::float32_t tmp;
		tmp.m_bits = detail::halfToFloat(m_value.m_bits);
		return tmp.m_float;
	}

	template<typename T>
	LIBRAPID_NODISCARD half::operator T() const noexcept {
		return static_cast<T>(static_cast<float>(*this));
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator+=(const half &rhs) noexcept {
		m_value.m_bits = detail::halfAdd(m_value.m_bits, rhs.m_value.m_bits);
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator-=(const half &rhs) noexcept {
		m_value.m_bits = detail::halfSub(m_value.m_bits, rhs.m_value.m_bits);
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator*=(const half &rhs) noexcept {
		m_value.m_bits = detail::halfMul(m_value.m_bits, rhs.m_value.m_bits);
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator/=(const half &rhs) noexcept {
		*this = static_cast<float>(*this) / static_cast<float>(rhs);
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator--() noexcept {
		*this -= half::fromBits(static_cast<uint16_t>(0x3c00));
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half half::operator--(int) noexcept {
		half tmp(*this);
		tmp -= half::fromBits(static_cast<uint16_t>(0x3c00));
		return tmp;
	}

	LIBRAPID_ALWAYS_INLINE half &half::operator++() noexcept {
		*this += half::fromBits(static_cast<uint16_t>(0x3c00));
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE half half::operator++(int) noexcept {
		half tmp(*this);
		tmp += half::fromBits(static_cast<uint16_t>(0x3c00));
		return tmp;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half half::operator-() const noexcept {
		return half::fromBits((m_value.m_bits & 0x7fff) | (m_value.m_bits ^ 0x8000));
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half half::operator+() const noexcept {
		return *this;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::float16_t half::data() const noexcept {
		return m_value;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::float16_t &half::data() noexcept {
		return m_value;
	}

	std::string half::str(const std::string &format) const {
		return fmt::format(format, static_cast<float>(*this));
	}

	//	inline half half::infinity			   = half::fromBits(static_cast<uint16_t>(0x7c00));
	//	inline half half::max				   = half::fromBits(static_cast<uint16_t>(0x7bff));
	//	inline half half::maxSubnormal		   = half::fromBits(static_cast<uint16_t>(0x3ff));
	//	inline half half::min				   = half::fromBits(static_cast<uint16_t>(0xfbff));
	//	inline half half::minPositive		   = half::fromBits(static_cast<uint16_t>(0x400));
	//	inline half half::minPositiveSubnormal = half::fromBits(static_cast<uint16_t>(0x1));
	//	inline half half::nan				   = half::fromBits(static_cast<uint16_t>(0x7e00));
	//	inline half half::negativeInfinity	   = half::fromBits(static_cast<uint16_t>(0xfc00));
	//	inline half half::epsilon			   = half::fromBits(static_cast<uint16_t>(0x1400));
	//
	//	inline half half::one		   = half::fromBits(static_cast<uint16_t>(0x3c00));
	//	inline half half::negativeOne  = half::fromBits(static_cast<uint16_t>(0x4000));
	//	inline half half::two		   = half::fromBits(static_cast<uint16_t>(0x4000));
	//	inline half half::negativeTwo  = half::fromBits(static_cast<uint16_t>(0xc000));
	//	inline half half::half_		   = half::fromBits(static_cast<uint16_t>(0x3800));
	//	inline half half::negativeHalf = half::fromBits(static_cast<uint16_t>(0x3b00));
	//	inline half half::zero		   = half::fromBits(static_cast<uint16_t>(0x0));
	//	inline half half::negativeZero = half::fromBits(static_cast<uint16_t>(0x8000));
	//	inline half half::e			   = half::fromBits(static_cast<uint16_t>(0x4170));
	//	inline half half::pi		   = half::fromBits(static_cast<uint16_t>(0x4248));

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator+(const half &lhs,
															 const half &rhs) noexcept {
		half tmp(lhs);
		tmp += rhs;
		return tmp;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator-(const half &lhs,
															 const half &rhs) noexcept {
		half tmp(lhs);
		tmp -= rhs;
		return tmp;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator*(const half &lhs,
															 const half &rhs) noexcept {
		half tmp(lhs);
		tmp *= rhs;
		return tmp;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE half operator/(const half &lhs,
															 const half &rhs) noexcept {
		half tmp(lhs);
		tmp /= rhs;
		return tmp;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator<(const half &lhs,
															 const half &rhs) noexcept {
		auto const &l_ieee = lhs.data().m_ieee;
		auto const &r_ieee = rhs.data().m_ieee;

		if (l_ieee.m_sign == 1) {
			if (r_ieee.m_sign == 0) return true;
			if (l_ieee.m_exp > r_ieee.m_exp) return true;
			if (l_ieee.m_exp < r_ieee.m_exp) return false;
			if (l_ieee.m_frac > r_ieee.m_frac) return true;
			return false;
		}

		if (r_ieee.m_sign == 1) return false;
		if (l_ieee.m_exp > r_ieee.m_exp) return false;
		if (l_ieee.m_exp < r_ieee.m_exp) return true;
		if (l_ieee.m_frac >= r_ieee.m_frac) return false;
		return true;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator==(const half &lhs,
															  const half &rhs) noexcept {
		return lhs.data().m_bits == rhs.data().m_bits;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator!=(const half &lhs,
															  const half &rhs) noexcept {
		return lhs.data().m_bits != rhs.data().m_bits;
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator<=(const half &lhs,
															  const half &rhs) noexcept {
		return (lhs < rhs) || (lhs == rhs);
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator>(const half &lhs,
															 const half &rhs) noexcept {
		return !(lhs <= rhs);
	}

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool operator>=(const half &lhs,
															  const half &rhs) noexcept {
		return !(lhs < rhs);
	}

	namespace typetraits {
		template<>
		struct TypeInfo<half> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar							   = half;
			using Packet							   = std::false_type;
			using Backend							   = backend::CPU;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr char name[]			   = "__half";
			static constexpr bool supportsArithmetic   = true;
			static constexpr bool supportsLogical	   = true;
			static constexpr bool supportsBinary	   = false;
			static constexpr bool allowVectorisation   = false;

#if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_16F;
			static constexpr int64_t cudaPacketWidth = 1;
#endif

			static constexpr bool canAlign	= true;
			static constexpr bool canMemcpy = true;

			// LIMIT_IMPL(min) { return half::min; }
			// LIMIT_IMPL(max) { return half::max; }
			// LIMIT_IMPL(epsilon) { return half::epsilon; }
			// LIMIT_IMPL(roundError) { return half::epsilon * static_cast<half>(0.5); }
			// LIMIT_IMPL(denormMin) { return half::minPositiveSubnormal; }
			// LIMIT_IMPL(infinity) { return half::infinity; }
			// LIMIT_IMPL(quietNaN) { return half::nan; }
			// LIMIT_IMPL(signalingNaN) { return half::nan; }

			LIMIT_IMPL(infinity) { return half::fromBits(static_cast<uint16_t>(0x7c00)); }
			LIMIT_IMPL(max) { return half::fromBits(static_cast<uint16_t>(0x7bff)); }
			LIMIT_IMPL(maxSubnormal) { return half::fromBits(static_cast<uint16_t>(0x3ff)); }
			LIMIT_IMPL(min) { return half::fromBits(static_cast<uint16_t>(0xfbff)); }
			LIMIT_IMPL(minPositive) { return half::fromBits(static_cast<uint16_t>(0x400)); }
			LIMIT_IMPL(minPositiveSubnormal) { return half::fromBits(static_cast<uint16_t>(0x1)); }
			LIMIT_IMPL(nan) { return half::fromBits(static_cast<uint16_t>(0x7e00)); }
			LIMIT_IMPL(negativeInfinity) { return half::fromBits(static_cast<uint16_t>(0xfc00)); }
			LIMIT_IMPL(epsilon) { return half::fromBits(static_cast<uint16_t>(0x1400)); }

			LIMIT_IMPL(one) { return half::fromBits(static_cast<uint16_t>(0x3c00)); }
			LIMIT_IMPL(negativeOne) { return half::fromBits(static_cast<uint16_t>(0x4000)); }
			LIMIT_IMPL(two) { return half::fromBits(static_cast<uint16_t>(0x4000)); }
			LIMIT_IMPL(negativeTwo) { return half::fromBits(static_cast<uint16_t>(0xc000)); }
			LIMIT_IMPL(half_) { return half::fromBits(static_cast<uint16_t>(0x3800)); }
			LIMIT_IMPL(negativeHalf) { return half::fromBits(static_cast<uint16_t>(0x3b00)); }
			LIMIT_IMPL(zero) { return half::fromBits(static_cast<uint16_t>(0x0)); }
			LIMIT_IMPL(negativeZero) { return half::fromBits(static_cast<uint16_t>(0x8000)); }
			LIMIT_IMPL(e) { return half::fromBits(static_cast<uint16_t>(0x4170)); }
			LIMIT_IMPL(pi) { return half::fromBits(static_cast<uint16_t>(0x4248)); }
		};
	} // namespace typetraits
} // namespace librapid

LIBRAPID_SIMPLE_IO_IMPL_NO_TEMPLATE(librapid::half);

#endif // LIBRAPID_MATH_HALF_HPP