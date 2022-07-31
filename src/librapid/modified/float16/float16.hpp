#pragma once

//
// inspired by:
// https://github.com/acgessler/half_float
// https://github.com/x448/float16
//
#include <cstdint>
#include <limits>
#include <iostream>
#include <cmath>
#include <bitset>
#include <type_traits>

namespace librapid {
	namespace half_private {
		constexpr inline std::uint32_t _uint32_sels(std::uint32_t test, std::uint32_t a,
													std::uint32_t b) noexcept {
			const std::uint32_t mask   = (((std::int32_t)test) >> 31);
			const std::uint32_t sel_a  = (a & mask);
			const std::uint32_t sel_b  = (b & ~mask);
			const std::uint32_t result = (sel_a | sel_b);
			return (result);
		}

		constexpr inline std::uint32_t _uint32_selb(std::uint32_t mask, std::uint32_t a,
													std::uint32_t b) noexcept {
			const std::uint32_t sel_a  = (a & mask);
			const std::uint32_t sel_b  = (b & ~mask);
			const std::uint32_t result = (sel_a | sel_b);
			return (result);
		}

		constexpr inline std::uint16_t _uint16_sels(std::uint16_t test, std::uint16_t a,
													std::uint16_t b) noexcept {
			const std::uint16_t mask   = (((int16_t)test) >> 15);
			const std::uint16_t sel_a  = (a & mask);
			const std::uint16_t sel_b  = (b & ~mask);
			const std::uint16_t result = (sel_a | sel_b);
			return (result);
		}

		constexpr inline std::uint32_t _uint32_cntlz(std::uint32_t x) noexcept {
#if defined(LIBRAPID_GNU_CXX)
			std::uint32_t is_x_nez_msb = (-x);
			std::uint32_t nlz		   = __builtin_clz(x);
			std::uint32_t result	   = _uint32_sels(is_x_nez_msb, nlz, 0x00000020);
			return (result);
#else
			const std::uint32_t x0	= (x >> 1);
			const std::uint32_t x1	= (x | x0);
			const std::uint32_t x2	= (x1 >> 2);
			const std::uint32_t x3	= (x1 | x2);
			const std::uint32_t x4	= (x3 >> 4);
			const std::uint32_t x5	= (x3 | x4);
			const std::uint32_t x6	= (x5 >> 8);
			const std::uint32_t x7	= (x5 | x6);
			const std::uint32_t x8	= (x7 >> 16);
			const std::uint32_t x9	= (x7 | x8);
			const std::uint32_t xA	= (~x9);
			const std::uint32_t xB	= (xA >> 1);
			const std::uint32_t xC	= (xB & 0x55555555);
			const std::uint32_t xD	= (xA - xC);
			const std::uint32_t xE	= (xD & 0x33333333);
			const std::uint32_t xF	= (xD >> 2);
			const std::uint32_t x10 = (xF & 0x33333333);
			const std::uint32_t x11 = (xE + x10);
			const std::uint32_t x12 = (x11 >> 4);
			const std::uint32_t x13 = (x11 + x12);
			const std::uint32_t x14 = (x13 & 0x0f0f0f0f);
			const std::uint32_t x15 = (x14 >> 8);
			const std::uint32_t x16 = (x14 + x15);
			const std::uint32_t x17 = (x16 >> 16);
			const std::uint32_t x18 = (x16 + x17);
			const std::uint32_t x19 = (x18 & 0x0000003f);
			return (x19);
#endif
		}

		constexpr inline std::uint16_t _uint16_cntlz(std::uint16_t x) noexcept {
#if defined(LIBRAPID_GNU_CXX)
			std::uint16_t nlz32 = (std::uint16_t)_uint32_cntlz((std::uint32_t)x);
			std::uint32_t nlz	= (nlz32 - 16);
			return (nlz);
#else
			const std::uint16_t x0	= (x >> 1);
			const std::uint16_t x1	= (x | x0);
			const std::uint16_t x2	= (x1 >> 2);
			const std::uint16_t x3	= (x1 | x2);
			const std::uint16_t x4	= (x3 >> 4);
			const std::uint16_t x5	= (x3 | x4);
			const std::uint16_t x6	= (x5 >> 8);
			const std::uint16_t x7	= (x5 | x6);
			const std::uint16_t x8	= (~x7);
			const std::uint16_t x9	= ((x8 >> 1) & 0x5555);
			const std::uint16_t xA	= (x8 - x9);
			const std::uint16_t xB	= (xA & 0x3333);
			const std::uint16_t xC	= ((xA >> 2) & 0x3333);
			const std::uint16_t xD	= (xB + xC);
			const std::uint16_t xE	= (xD >> 4);
			const std::uint16_t xF	= ((xD + xE) & 0x0f0f);
			const std::uint16_t x10 = (xF >> 8);
			const std::uint16_t x11 = ((xF + x10) & 0x001f);
			return (x11);
#endif
		}
	} // namespace half_private

	constexpr inline std::uint16_t float_to_half(std::uint32_t f) noexcept {
		const std::uint32_t one						 = (0x00000001);
		const std::uint32_t f_s_mask				 = (0x80000000);
		const std::uint32_t f_e_mask				 = (0x7f800000);
		const std::uint32_t f_m_mask				 = (0x007fffff);
		const std::uint32_t f_m_hidden_bit			 = (0x00800000);
		const std::uint32_t f_m_round_bit			 = (0x00001000);
		const std::uint32_t f_snan_mask				 = (0x7fc00000);
		const std::uint32_t f_e_pos					 = (0x00000017);
		const std::uint32_t h_e_pos					 = (0x0000000a);
		const std::uint32_t h_e_mask				 = (0x00007c00);
		const std::uint32_t h_snan_mask				 = (0x00007e00);
		const std::uint32_t h_e_mask_value			 = (0x0000001f);
		const std::uint32_t f_h_s_pos_offset		 = (0x00000010);
		const std::uint32_t f_h_bias_offset			 = (0x00000070);
		const std::uint32_t f_h_m_pos_offset		 = (0x0000000d);
		const std::uint32_t h_nan_min				 = (0x00007c01);
		const std::uint32_t f_h_e_biased_flag		 = (0x0000008f);
		const std::uint32_t f_s						 = (f & f_s_mask);
		const std::uint32_t f_e						 = (f & f_e_mask);
		const std::uint16_t h_s						 = (f_s >> f_h_s_pos_offset);
		const std::uint32_t f_m						 = (f & f_m_mask);
		const std::uint16_t f_e_amount				 = (f_e >> f_e_pos);
		const std::uint32_t f_e_half_bias			 = (f_e_amount - f_h_bias_offset);
		const std::uint32_t f_snan					 = (f & f_snan_mask);
		const std::uint32_t f_m_round_mask			 = (f_m & f_m_round_bit);
		const std::uint32_t f_m_round_offset		 = (f_m_round_mask << one);
		const std::uint32_t f_m_rounded				 = (f_m + f_m_round_offset);
		const std::uint32_t f_m_denorm_sa			 = (one - f_e_half_bias);
		const std::uint32_t f_m_with_hidden			 = (f_m_rounded | f_m_hidden_bit);
		const std::uint32_t f_m_denorm				 = (f_m_with_hidden >> f_m_denorm_sa);
		const std::uint32_t h_m_denorm				 = (f_m_denorm >> f_h_m_pos_offset);
		const std::uint32_t f_m_rounded_overflow	 = (f_m_rounded & f_m_hidden_bit);
		const std::uint32_t m_nan					 = (f_m >> f_h_m_pos_offset);
		const std::uint32_t h_em_nan				 = (h_e_mask | m_nan);
		const std::uint32_t h_e_norm_overflow_offset = (f_e_half_bias + 1);
		const std::uint32_t h_e_norm_overflow		 = (h_e_norm_overflow_offset << h_e_pos);
		const std::uint32_t h_e_norm				 = (f_e_half_bias << h_e_pos);
		const std::uint32_t h_m_norm				 = (f_m_rounded >> f_h_m_pos_offset);
		const std::uint32_t h_em_norm				 = (h_e_norm | h_m_norm);
		const std::uint32_t is_h_ndenorm_msb		 = (f_h_bias_offset - f_e_amount);
		const std::uint32_t is_f_e_flagged_msb		 = (f_h_e_biased_flag - f_e_half_bias);
		const std::uint32_t is_h_denorm_msb			 = (~is_h_ndenorm_msb);
		const std::uint32_t is_f_m_eqz_msb			 = (f_m - 1);
		const std::uint32_t is_h_nan_eqz_msb		 = (m_nan - 1);
		const std::uint32_t is_f_inf_msb			 = (is_f_e_flagged_msb & is_f_m_eqz_msb);
		const std::uint32_t is_f_nan_underflow_msb	 = (is_f_e_flagged_msb & is_h_nan_eqz_msb);
		const std::uint32_t is_e_overflow_msb		 = (h_e_mask_value - f_e_half_bias);
		const std::uint32_t is_h_inf_msb			 = (is_e_overflow_msb | is_f_inf_msb);
		const std::uint32_t is_f_nsnan_msb			 = (f_snan - f_snan_mask);
		const std::uint32_t is_m_norm_overflow_msb	 = (-f_m_rounded_overflow);
		const std::uint32_t is_f_snan_msb			 = (~is_f_nsnan_msb);
		const std::uint32_t h_em_overflow_result =
		  half_private::_uint32_sels(is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm);
		const std::uint32_t h_em_nan_result =
		  half_private::_uint32_sels(is_f_e_flagged_msb, h_em_nan, h_em_overflow_result);
		const std::uint32_t h_em_nan_underflow_result =
		  half_private::_uint32_sels(is_f_nan_underflow_msb, h_nan_min, h_em_nan_result);
		const std::uint32_t h_em_inf_result =
		  half_private::_uint32_sels(is_h_inf_msb, h_e_mask, h_em_nan_underflow_result);
		const std::uint32_t h_em_denorm_result =
		  half_private::_uint32_sels(is_h_denorm_msb, h_m_denorm, h_em_inf_result);
		const std::uint32_t h_em_snan_result =
		  half_private::_uint32_sels(is_f_snan_msb, h_snan_mask, h_em_denorm_result);
		const std::uint32_t h_result = (h_s | h_em_snan_result);
		return (std::uint16_t)(h_result);
	}

	constexpr inline std::uint32_t half_to_float(std::uint16_t h) noexcept {
		const std::uint32_t h_e_mask			 = (0x00007c00);
		const std::uint32_t h_m_mask			 = (0x000003ff);
		const std::uint32_t h_s_mask			 = (0x00008000);
		const std::uint32_t h_f_s_pos_offset	 = (0x00000010);
		const std::uint32_t h_f_e_pos_offset	 = (0x0000000d);
		const std::uint32_t h_f_bias_offset		 = (0x0001c000);
		const std::uint32_t f_e_mask			 = (0x7f800000);
		const std::uint32_t f_m_mask			 = (0x007fffff);
		const std::uint32_t h_f_e_denorm_bias	 = (0x0000007e);
		const std::uint32_t h_f_m_denorm_sa_bias = (0x00000008);
		const std::uint32_t f_e_pos				 = (0x00000017);
		const std::uint32_t h_e_mask_minus_one	 = (0x00007bff);
		const std::uint32_t h_e					 = (h & h_e_mask);
		const std::uint32_t h_m					 = (h & h_m_mask);
		const std::uint32_t h_s					 = (h & h_s_mask);
		const std::uint32_t h_e_f_bias			 = (h_e + h_f_bias_offset);
		const std::uint32_t h_m_nlz				 = half_private::_uint32_cntlz(h_m);
		const std::uint32_t f_s					 = (h_s << h_f_s_pos_offset);
		const std::uint32_t f_e					 = (h_e_f_bias << h_f_e_pos_offset);
		const std::uint32_t f_m					 = (h_m << h_f_e_pos_offset);
		const std::uint32_t f_em				 = (f_e | f_m);
		const std::uint32_t h_f_m_sa			 = (h_m_nlz - h_f_m_denorm_sa_bias);
		const std::uint32_t f_e_denorm_unpacked	 = (h_f_e_denorm_bias - h_f_m_sa);
		const std::uint32_t h_f_m				 = (h_m << h_f_m_sa);
		const std::uint32_t f_m_denorm			 = (h_f_m & f_m_mask);
		const std::uint32_t f_e_denorm			 = (f_e_denorm_unpacked << f_e_pos);
		const std::uint32_t f_em_denorm			 = (f_e_denorm | f_m_denorm);
		const std::uint32_t f_em_nan			 = (f_e_mask | f_m);
		const std::uint32_t is_e_eqz_msb		 = (h_e - 1);
		const std::uint32_t is_m_nez_msb		 = (-h_m);
		const std::uint32_t is_e_flagged_msb	 = (h_e_mask_minus_one - h_e);
		const std::uint32_t is_zero_msb			 = (is_e_eqz_msb & ~is_m_nez_msb);
		const std::uint32_t is_inf_msb			 = (is_e_flagged_msb & ~is_m_nez_msb);
		const std::uint32_t is_denorm_msb		 = (is_m_nez_msb & is_e_eqz_msb);
		const std::uint32_t is_nan_msb			 = (is_e_flagged_msb & is_m_nez_msb);
		const std::uint32_t is_zero				 = (((std::int32_t)is_zero_msb) >> 31);
		const std::uint32_t f_zero_result		 = (f_em & ~is_zero);
		const std::uint32_t f_denorm_result =
		  half_private::_uint32_sels(is_denorm_msb, f_em_denorm, f_zero_result);
		const std::uint32_t f_inf_result =
		  half_private::_uint32_sels(is_inf_msb, f_e_mask, f_denorm_result);
		const std::uint32_t f_nan_result =
		  half_private::_uint32_sels(is_nan_msb, f_em_nan, f_inf_result);
		const std::uint32_t f_result = (f_s | f_nan_result);
		return (f_result);
	}

	constexpr inline std::uint16_t half_add(std::uint16_t x, std::uint16_t y) noexcept {
		constexpr std::uint16_t one				   = (0x0001);
		constexpr std::uint16_t msb_to_lsb_sa	   = (0x000f);
		constexpr std::uint16_t h_s_mask		   = (0x8000);
		constexpr std::uint16_t h_e_mask		   = (0x7c00);
		constexpr std::uint16_t h_m_mask		   = (0x03ff);
		constexpr std::uint16_t h_m_msb_mask	   = (0x2000);
		constexpr std::uint16_t h_m_msb_sa		   = (0x000d);
		constexpr std::uint16_t h_m_hidden		   = (0x0400);
		constexpr std::uint16_t h_e_pos			   = (0x000a);
		constexpr std::uint16_t h_e_bias_minus_one = (0x000e);
		constexpr std::uint16_t h_m_grs_carry	   = (0x4000);
		constexpr std::uint16_t h_m_grs_carry_pos  = (0x000e);
		constexpr std::uint16_t h_grs_size		   = (0x0003);
		constexpr std::uint16_t h_snan			   = (0xfe00);
		constexpr std::uint16_t h_e_mask_minus_one = (0x7bff);
		const std::uint16_t h_grs_round_carry	   = (one << h_grs_size);
		const std::uint16_t h_grs_round_mask	   = (h_grs_round_carry - one);
		const std::uint16_t x_e					   = (x & h_e_mask);
		const std::uint16_t y_e					   = (y & h_e_mask);
		const std::uint16_t is_y_e_larger_msb	   = (x_e - y_e);
		const std::uint16_t a				  = half_private::_uint16_sels(is_y_e_larger_msb, y, x);
		const std::uint16_t a_s				  = (a & h_s_mask);
		const std::uint16_t a_e				  = (a & h_e_mask);
		const std::uint16_t a_m_no_hidden_bit = (a & h_m_mask);
		const std::uint16_t a_em_no_hidden_bit = (a_e | a_m_no_hidden_bit);
		const std::uint16_t b				  = half_private::_uint16_sels(is_y_e_larger_msb, x, y);
		const std::uint16_t b_s				  = (b & h_s_mask);
		const std::uint16_t b_e				  = (b & h_e_mask);
		const std::uint16_t b_m_no_hidden_bit = (b & h_m_mask);
		const std::uint16_t b_em_no_hidden_bit	  = (b_e | b_m_no_hidden_bit);
		const std::uint16_t is_diff_sign_msb	  = (a_s ^ b_s);
		const std::uint16_t is_a_inf_msb		  = (h_e_mask_minus_one - a_em_no_hidden_bit);
		const std::uint16_t is_b_inf_msb		  = (h_e_mask_minus_one - b_em_no_hidden_bit);
		const std::uint16_t is_undenorm_msb		  = (a_e - 1);
		const std::uint16_t is_undenorm			  = (((int16_t)is_undenorm_msb) >> 15);
		const std::uint16_t is_both_inf_msb		  = (is_a_inf_msb & is_b_inf_msb);
		const std::uint16_t is_invalid_inf_op_msb = (is_both_inf_msb & b_s);
		const std::uint16_t is_a_e_nez_msb		  = (-a_e);
		const std::uint16_t is_b_e_nez_msb		  = (-b_e);
		const std::uint16_t is_a_e_nez			  = (((int16_t)is_a_e_nez_msb) >> 15);
		const std::uint16_t is_b_e_nez			  = (((int16_t)is_b_e_nez_msb) >> 15);
		const std::uint16_t a_m_hidden_bit		  = (is_a_e_nez & h_m_hidden);
		const std::uint16_t b_m_hidden_bit		  = (is_b_e_nez & h_m_hidden);
		const std::uint16_t a_m_no_grs			  = (a_m_no_hidden_bit | a_m_hidden_bit);
		const std::uint16_t b_m_no_grs			  = (b_m_no_hidden_bit | b_m_hidden_bit);
		const std::uint16_t diff_e				  = (a_e - b_e);
		const std::uint16_t a_e_unbias			  = (a_e - h_e_bias_minus_one);
		const std::uint16_t a_m					  = (a_m_no_grs << h_grs_size);
		const std::uint16_t a_e_biased			  = (a_e >> h_e_pos);
		const std::uint16_t m_sa_unbias			  = (a_e_unbias >> h_e_pos);
		const std::uint16_t m_sa_default		  = (diff_e >> h_e_pos);
		const std::uint16_t m_sa_unbias_mask	  = (is_a_e_nez_msb & ~is_b_e_nez_msb);
		const std::uint16_t m_sa =
		  half_private::_uint16_sels(m_sa_unbias_mask, m_sa_unbias, m_sa_default);
		const std::uint16_t b_m_no_sticky	  = (b_m_no_grs << h_grs_size);
		const std::uint16_t sh_m			  = (b_m_no_sticky >> m_sa);
		const std::uint16_t sticky_overflow	  = (one << m_sa);
		const std::uint16_t sticky_mask		  = (sticky_overflow - 1);
		const std::uint16_t sticky_collect	  = (b_m_no_sticky & sticky_mask);
		const std::uint16_t is_sticky_set_msb = (-sticky_collect);
		const std::uint16_t sticky			  = (is_sticky_set_msb >> msb_to_lsb_sa);
		const std::uint16_t b_m				  = (sh_m | sticky);
		const std::uint16_t is_c_m_ab_pos_msb = (b_m - a_m);
		const std::uint16_t c_inf			  = (a_s | h_e_mask);
		const std::uint16_t c_m_sum			  = (a_m + b_m);
		const std::uint16_t c_m_diff_ab		  = (a_m - b_m);
		const std::uint16_t c_m_diff_ba		  = (b_m - a_m);
		const std::uint16_t c_m_smag_diff =
		  half_private::_uint16_sels(is_c_m_ab_pos_msb, c_m_diff_ab, c_m_diff_ba);
		const std::uint16_t c_s_diff = half_private::_uint16_sels(is_c_m_ab_pos_msb, a_s, b_s);
		const std::uint16_t c_s		 = half_private::_uint16_sels(is_diff_sign_msb, c_s_diff, a_s);
		const std::uint16_t c_m_smag_diff_nlz  = half_private::_uint16_cntlz(c_m_smag_diff);
		const std::uint16_t diff_norm_sa	   = (c_m_smag_diff_nlz - one);
		const std::uint16_t is_diff_denorm_msb = (a_e_biased - diff_norm_sa);
		const std::uint16_t is_diff_denorm	   = (((int16_t)is_diff_denorm_msb) >> 15);
		const std::uint16_t is_a_or_b_norm_msb = (-a_e_biased);
		const std::uint16_t diff_denorm_sa	   = (a_e_biased - 1);
		const std::uint16_t c_m_diff_denorm	   = (c_m_smag_diff << diff_denorm_sa);
		const std::uint16_t c_m_diff_norm	   = (c_m_smag_diff << diff_norm_sa);
		const std::uint16_t c_e_diff_norm	   = (a_e_biased - diff_norm_sa);
		const std::uint16_t c_m_diff_ab_norm =
		  half_private::_uint16_sels(is_diff_denorm_msb, c_m_diff_denorm, c_m_diff_norm);
		const std::uint16_t c_e_diff_ab_norm = (c_e_diff_norm & ~is_diff_denorm);
		const std::uint16_t c_m_diff =
		  half_private::_uint16_sels(is_a_or_b_norm_msb, c_m_diff_ab_norm, c_m_smag_diff);
		const std::uint16_t c_e_diff =
		  half_private::_uint16_sels(is_a_or_b_norm_msb, c_e_diff_ab_norm, a_e_biased);
		const std::uint16_t is_diff_eqz_msb			 = (c_m_diff - 1);
		const std::uint16_t is_diff_exactly_zero_msb = (is_diff_sign_msb & is_diff_eqz_msb);
		const std::uint16_t is_diff_exactly_zero	 = (((int16_t)is_diff_exactly_zero_msb) >> 15);
		const std::uint16_t c_m_added =
		  half_private::_uint16_sels(is_diff_sign_msb, c_m_diff, c_m_sum);
		const std::uint16_t c_e_added =
		  half_private::_uint16_sels(is_diff_sign_msb, c_e_diff, a_e_biased);
		const std::uint16_t c_m_carry		  = (c_m_added & h_m_grs_carry);
		const std::uint16_t is_c_m_carry_msb  = (-c_m_carry);
		const std::uint16_t c_e_hidden_offset = ((c_m_added & h_m_grs_carry) >> h_m_grs_carry_pos);
		const std::uint16_t c_m_sub_hidden	  = (c_m_added >> one);
		const std::uint16_t c_m_no_hidden =
		  half_private::_uint16_sels(is_c_m_carry_msb, c_m_sub_hidden, c_m_added);
		const std::uint16_t c_e_no_hidden	   = (c_e_added + c_e_hidden_offset);
		const std::uint16_t c_m_no_hidden_msb  = (c_m_no_hidden & h_m_msb_mask);
		const std::uint16_t undenorm_m_msb_odd = (c_m_no_hidden_msb >> h_m_msb_sa);
		const std::uint16_t undenorm_fix_e	   = (is_undenorm & undenorm_m_msb_odd);
		const std::uint16_t c_e_fixed		   = (c_e_no_hidden + undenorm_fix_e);
		const std::uint16_t c_m_round_amount   = (c_m_no_hidden & h_grs_round_mask);
		const std::uint16_t c_m_rounded		   = (c_m_no_hidden + c_m_round_amount);
		const std::uint16_t c_m_round_overflow =
		  ((c_m_rounded & h_m_grs_carry) >> h_m_grs_carry_pos);
		const std::uint16_t c_e_rounded = (c_e_fixed + c_m_round_overflow);
		const std::uint16_t c_m_no_grs	= ((c_m_rounded >> h_grs_size) & h_m_mask);
		const std::uint16_t c_e			= (c_e_rounded << h_e_pos);
		const std::uint16_t c_em		= (c_e | c_m_no_grs);
		const std::uint16_t c_normal	= (c_s | c_em);
		const std::uint16_t c_inf_result =
		  half_private::_uint16_sels(is_a_inf_msb, c_inf, c_normal);
		const std::uint16_t c_zero_result = (c_inf_result & ~is_diff_exactly_zero);
		const std::uint16_t c_result =
		  half_private::_uint16_sels(is_invalid_inf_op_msb, h_snan, c_zero_result);
		return (c_result);
	}

	constexpr inline std::uint16_t half_mul(std::uint16_t x, std::uint16_t y) noexcept {
		const std::uint32_t one						= (0x00000001);
		const std::uint32_t h_s_mask				= (0x00008000);
		const std::uint32_t h_e_mask				= (0x00007c00);
		const std::uint32_t h_m_mask				= (0x000003ff);
		const std::uint32_t h_m_hidden				= (0x00000400);
		const std::uint32_t h_e_pos					= (0x0000000a);
		const std::uint32_t h_e_bias				= (0x0000000f);
		const std::uint32_t h_m_bit_count			= (0x0000000a);
		const std::uint32_t h_m_bit_half_count		= (0x00000005);
		const std::uint32_t h_nan_min				= (0x00007c01);
		const std::uint32_t h_e_mask_minus_one		= (0x00007bff);
		const std::uint32_t h_snan					= (0x0000fe00);
		const std::uint32_t m_round_overflow_bit	= (0x00000020);
		const std::uint32_t m_hidden_bit			= (0x00100000);
		const std::uint32_t a_s						= (x & h_s_mask);
		const std::uint32_t b_s						= (y & h_s_mask);
		const std::uint32_t c_s						= (a_s ^ b_s);
		const std::uint32_t x_e						= (x & h_e_mask);
		const std::uint32_t x_e_eqz_msb				= (x_e - 1);
		const std::uint32_t a						= half_private::_uint32_sels(x_e_eqz_msb, y, x);
		const std::uint32_t b						= half_private::_uint32_sels(x_e_eqz_msb, x, y);
		const std::uint32_t a_e						= (a & h_e_mask);
		const std::uint32_t b_e						= (b & h_e_mask);
		const std::uint32_t a_m						= (a & h_m_mask);
		const std::uint32_t b_m						= (b & h_m_mask);
		const std::uint32_t a_e_amount				= (a_e >> h_e_pos);
		const std::uint32_t b_e_amount				= (b_e >> h_e_pos);
		const std::uint32_t a_m_with_hidden			= (a_m | h_m_hidden);
		const std::uint32_t b_m_with_hidden			= (b_m | h_m_hidden);
		const std::uint32_t c_m_normal				= (a_m_with_hidden * b_m_with_hidden);
		const std::uint32_t c_m_denorm_biased		= (a_m_with_hidden * b_m);
		const std::uint32_t c_e_denorm_unbias_e		= (h_e_bias - a_e_amount);
		const std::uint32_t c_m_denorm_round_amount = (c_m_denorm_biased & h_m_mask);
		const std::uint32_t c_m_denorm_rounded		= (c_m_denorm_biased + c_m_denorm_round_amount);
		const std::uint32_t c_m_denorm_inplace		= (c_m_denorm_rounded >> h_m_bit_count);
		const std::uint32_t c_m_denorm_unbiased		= (c_m_denorm_inplace >> c_e_denorm_unbias_e);
		const std::uint32_t c_m_denorm				= (c_m_denorm_unbiased & h_m_mask);
		const std::uint32_t c_e_amount_biased		= (a_e_amount + b_e_amount);
		const std::uint32_t c_e_amount_unbiased		= (c_e_amount_biased - h_e_bias);
		const std::uint32_t is_c_e_unbiased_underflow = (((std::int32_t)c_e_amount_unbiased) >> 31);
		const std::uint32_t c_e_underflow_half_sa	  = (-c_e_amount_unbiased);
		const std::uint32_t c_e_underflow_sa		  = (c_e_underflow_half_sa << one);
		const std::uint32_t c_m_underflow			  = (c_m_normal >> c_e_underflow_sa);
		const std::uint32_t c_e_underflow_added =
		  (c_e_amount_unbiased & ~is_c_e_unbiased_underflow);
		const std::uint32_t c_m_underflow_added =
		  half_private::_uint32_selb(is_c_e_unbiased_underflow, c_m_underflow, c_m_normal);
		const std::uint32_t is_mul_overflow_test	 = (c_e_underflow_added & m_round_overflow_bit);
		const std::uint32_t is_mul_overflow_msb		 = (-is_mul_overflow_test);
		const std::uint32_t c_e_norm_radix_corrected = (c_e_underflow_added + 1);
		const std::uint32_t c_m_norm_radix_corrected = (c_m_underflow_added >> one);
		const std::uint32_t c_m_norm_hidden_bit		 = (c_m_norm_radix_corrected & m_hidden_bit);
		const std::uint32_t is_c_m_norm_no_hidden_msb = (c_m_norm_hidden_bit - 1);
		const std::uint32_t c_m_norm_lo		= (c_m_norm_radix_corrected >> h_m_bit_half_count);
		const std::uint32_t c_m_norm_lo_nlz = half_private::_uint16_cntlz(c_m_norm_lo);
		const std::uint32_t is_c_m_hidden_nunderflow_msb =
		  (c_m_norm_lo_nlz - c_e_norm_radix_corrected);
		const std::uint32_t is_c_m_hidden_underflow_msb = (~is_c_m_hidden_nunderflow_msb);
		const std::uint32_t is_c_m_hidden_underflow =
		  (((std::int32_t)is_c_m_hidden_underflow_msb) >> 31);
		const std::uint32_t c_m_hidden_underflow_normalized_sa = (c_m_norm_lo_nlz >> one);
		const std::uint32_t c_m_hidden_underflow_normalized =
		  (c_m_norm_radix_corrected << c_m_hidden_underflow_normalized_sa);
		const std::uint32_t c_m_hidden_normalized = (c_m_norm_radix_corrected << c_m_norm_lo_nlz);
		const std::uint32_t c_e_hidden_normalized = (c_e_norm_radix_corrected - c_m_norm_lo_nlz);
		const std::uint32_t c_e_hidden = (c_e_hidden_normalized & ~is_c_m_hidden_underflow);
		const std::uint32_t c_m_hidden = half_private::_uint32_sels(
		  is_c_m_hidden_underflow_msb, c_m_hidden_underflow_normalized, c_m_hidden_normalized);
		const std::uint32_t c_m_normalized = half_private::_uint32_sels(
		  is_c_m_norm_no_hidden_msb, c_m_hidden, c_m_norm_radix_corrected);
		const std::uint32_t c_e_normalized = half_private::_uint32_sels(
		  is_c_m_norm_no_hidden_msb, c_e_hidden, c_e_norm_radix_corrected);
		const std::uint32_t c_m_norm_round_amount  = (c_m_normalized & h_m_mask);
		const std::uint32_t c_m_norm_rounded	   = (c_m_normalized + c_m_norm_round_amount);
		const std::uint32_t is_round_overflow_test = (c_e_normalized & m_round_overflow_bit);
		const std::uint32_t is_round_overflow_msb  = (-is_round_overflow_test);
		const std::uint32_t c_m_norm_inplace	   = (c_m_norm_rounded >> h_m_bit_count);
		const std::uint32_t c_m					   = (c_m_norm_inplace & h_m_mask);
		const std::uint32_t c_e_norm_inplace	   = (c_e_normalized << h_e_pos);
		const std::uint32_t c_e					   = (c_e_norm_inplace & h_e_mask);
		const std::uint32_t c_em_nan			   = (h_e_mask | a_m);
		const std::uint32_t c_nan				   = (a_s | c_em_nan);
		const std::uint32_t c_denorm			   = (c_s | c_m_denorm);
		const std::uint32_t c_inf				   = (c_s | h_e_mask);
		const std::uint32_t c_em_norm			   = (c_e | c_m);
		const std::uint32_t is_a_e_flagged_msb	   = (h_e_mask_minus_one - a_e);
		const std::uint32_t is_b_e_flagged_msb	   = (h_e_mask_minus_one - b_e);
		const std::uint32_t is_a_e_eqz_msb		   = (a_e - 1);
		const std::uint32_t is_a_m_eqz_msb		   = (a_m - 1);
		const std::uint32_t is_b_e_eqz_msb		   = (b_e - 1);
		const std::uint32_t is_b_m_eqz_msb		   = (b_m - 1);
		const std::uint32_t is_b_eqz_msb		   = (is_b_e_eqz_msb & is_b_m_eqz_msb);
		const std::uint32_t is_a_eqz_msb		   = (is_a_e_eqz_msb & is_a_m_eqz_msb);
		const std::uint32_t is_c_nan_via_a_msb	   = (is_a_e_flagged_msb & ~is_b_e_flagged_msb);
		const std::uint32_t is_c_nan_via_b_msb	   = (is_b_e_flagged_msb & ~is_b_m_eqz_msb);
		const std::uint32_t is_c_nan_msb		   = (is_c_nan_via_a_msb | is_c_nan_via_b_msb);
		const std::uint32_t is_c_denorm_msb		   = (is_b_e_eqz_msb & ~is_a_e_flagged_msb);
		const std::uint32_t is_a_inf_msb		   = (is_a_e_flagged_msb & is_a_m_eqz_msb);
		const std::uint32_t is_c_snan_msb		   = (is_a_inf_msb & is_b_eqz_msb);
		const std::uint32_t is_c_nan_min_via_a_msb = (is_a_e_flagged_msb & is_b_eqz_msb);
		const std::uint32_t is_c_nan_min_via_b_msb = (is_b_e_flagged_msb & is_a_eqz_msb);
		const std::uint32_t is_c_nan_min_msb = (is_c_nan_min_via_a_msb | is_c_nan_min_via_b_msb);
		const std::uint32_t is_c_inf_msb	 = (is_a_e_flagged_msb | is_b_e_flagged_msb);
		const std::uint32_t is_overflow_msb	 = (is_round_overflow_msb | is_mul_overflow_msb);
		const std::uint32_t c_em_overflow_result =
		  half_private::_uint32_sels(is_overflow_msb, h_e_mask, c_em_norm);
		const std::uint32_t c_common_result = (c_s | c_em_overflow_result);
		const std::uint32_t c_zero_result =
		  half_private::_uint32_sels(is_b_eqz_msb, c_s, c_common_result);
		const std::uint32_t c_nan_result =
		  half_private::_uint32_sels(is_c_nan_msb, c_nan, c_zero_result);
		const std::uint32_t c_nan_min_result =
		  half_private::_uint32_sels(is_c_nan_min_msb, h_nan_min, c_nan_result);
		const std::uint32_t c_inf_result =
		  half_private::_uint32_sels(is_c_inf_msb, c_inf, c_nan_min_result);
		const std::uint32_t c_denorm_result =
		  half_private::_uint32_sels(is_c_denorm_msb, c_denorm, c_inf_result);
		const std::uint32_t c_result =
		  half_private::_uint32_sels(is_c_snan_msb, h_snan, c_denorm_result);
		return (std::uint16_t)(c_result);
	}

	constexpr inline std::uint16_t half_neg(std::uint16_t h) noexcept { return h ^ 0x8000; }

	constexpr inline std::uint16_t half_sub(std::uint16_t ha, std::uint16_t hb) noexcept {
		return half_add(ha, half_neg(hb));
	}

} // namespace librapid

namespace librapid::extended {
#ifdef LIBRAPID_DEBUG
	constexpr inline unsigned long const float16_debug_mode = 1;
#else
	constexpr inline unsigned long const float16_debug_mode = 0;
#endif

	namespace float16_t_private {

		union float16 {
			std::uint16_t bits_;
			struct {
				std::uint16_t frac_ : 10;
				std::uint16_t exp_ : 5;
				std::uint16_t sign_ : 1;
			} ieee_;
		};

		template<typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
		constexpr inline float16 to_float16(T val) noexcept {
			return float16 {static_cast<std::uint16_t>(val)};
		}

		inline std::ostream &operator<<(std::ostream &os, float16 const &f16) {
			os << std::bitset<1>(f16.ieee_.sign_) << " " << std::bitset<5>(f16.ieee_.exp_) << " "
			   << std::bitset<10>(f16.ieee_.frac_);
			return os;
		}

		union float32 {
			std::uint32_t bits_;
			struct {
				uint32_t frac_ : 23;
				uint32_t exp_ : 8;
				uint32_t sign_ : 1;
			} ieee_;
			float float_;
		};

		inline std::ostream &operator<<(std::ostream &os, float32 const &f32) {
			os << std::bitset<1>(f32.ieee_.sign_) << " " << std::bitset<8>(f32.ieee_.exp_) << " "
			   << std::bitset<23>(f32.ieee_.frac_) << " (" << f32.float_ << ")";
			return os;
		}

		inline float16 float32_to_float16(float input) noexcept {
			float32 f32;
			f32.float_ = input;
			float16 f16;
			f16.bits_ = librapid::float_to_half(f32.bits_);
			return f16;
		}

		inline float32 float16_to_float32(std::uint16_t input) noexcept {
			float32 f32;
			f32.bits_ = librapid::half_to_float(input);
			return f32;
		}

	} // namespace float16_t_private

	struct float16_t {
		float16_t_private::float16 data_;

		inline float16_t() noexcept					 = default;
		inline float16_t(float16_t const &) noexcept = default;
		inline float16_t(float16_t &&) noexcept		 = default;
		inline float16_t(float other) noexcept :
				data_ {float16_t_private::float32_to_float16(other)} {}
		inline float16_t(std::uint16_t bits) noexcept : data_ {bits} {}

		template<typename T>
		inline float16_t(T val) : float16_t((float)val) {}

		inline float16_t &operator=(float16_t const &) noexcept = default;
		inline float16_t &operator=(float16_t &&) noexcept		= default;
		inline float16_t &operator=(std::uint16_t bits) noexcept {
			data_.bits_ = bits;
			return *this;
		}
		inline float16_t &operator=(float other) noexcept {
			data_ = float16_t_private::float32_to_float16(other);
			return *this;
		}

		explicit inline operator float() const noexcept {
			auto f32 = float16_t_private::float16_to_float32(data_.bits_);
			return f32.float_;
		}

		template<typename T>
		explicit inline operator T() const noexcept {
			return (T)(float)*this;
		}

		inline float16_t &operator+=(float16_t v) noexcept {
			data_.bits_ = librapid::half_add(data_.bits_, v.data_.bits_);
			return *this;
		}

		inline float16_t &operator-=(float16_t v) noexcept {
			data_.bits_ = librapid::half_sub(data_.bits_, v.data_.bits_);
			return *this;
		}

		inline float16_t &operator*=(float16_t v) noexcept {
			data_.bits_ = librapid::half_mul(data_.bits_, v.data_.bits_);
			return *this;
		}

		inline float16_t &operator/=(float16_t v) noexcept {
			*this = float(*this) / float(v);
			return *this;
		}

		inline float16_t &operator+=(float v) noexcept {
			auto f16	= float16_t_private::float32_to_float16(v);
			data_.bits_ = librapid::half_add(data_.bits_, f16.bits_);
			return *this;
		}

		inline float16_t &operator-=(float v) noexcept {
			auto f16	= float16_t_private::float32_to_float16(v);
			data_.bits_ = librapid::half_sub(data_.bits_, f16.bits_);
			return *this;
		}

		inline float16_t &operator*=(float v) noexcept {
			auto f16	= float16_t_private::float32_to_float16(v);
			data_.bits_ = librapid::half_mul(data_.bits_, f16.bits_);
			return *this;
		}

		inline float16_t &operator/=(float v) noexcept {
			*this = float(*this) / v;
			return *this;
		}

		inline float16_t operator--() noexcept //--f
		{
			*this -= float16_t {static_cast<std::uint16_t>(0x3c00)};
			return *this;
		}

		inline const float16_t operator--(int) noexcept // f--
		{
			float16_t ans {*this};
			*this -= float16_t {static_cast<std::uint16_t>(0x3c00)};
			return ans;
		}

		inline float16_t operator++() noexcept //++f
		{
			*this += float16_t {static_cast<std::uint16_t>(0x3c00)};
			return *this;
		}

		inline const float16_t operator++(int) noexcept // f++
		{
			float16_t ans {*this};
			*this += float16_t {static_cast<std::uint16_t>(0x3c00)};
			return ans;
		}

		inline float16_t operator-() const noexcept {
			return float16_t {
			  static_cast<std::uint16_t>((data_.bits_ & 0x7fff) | (data_.bits_ ^ 0x8000))};
		}

		inline float16_t operator+() const noexcept { return *this; }

	}; // struct float16_t

	inline float16_t fp16_infinity {static_cast<std::uint16_t>(0x7c00)};
	inline float16_t fp16_max {static_cast<std::uint16_t>(0x7bff)};			 // 65504
	inline float16_t fp16_max_subnormal {static_cast<std::uint16_t>(0x3ff)}; // 0.00006097
	inline float16_t fp16_min {static_cast<std::uint16_t>(0xfbff)};
	inline float16_t fp16_min_positive {static_cast<std::uint16_t>(0x400)};
	inline float16_t fp16_min_positive_subnormal {static_cast<std::uint16_t>(0x1)};
	inline float16_t fp16_nan {static_cast<std::uint16_t>(0x7e00)};
	inline float16_t fp16_infinity_negative {static_cast<std::uint16_t>(0xfc00)};
	inline float16_t fp16_epsilon {static_cast<std::uint16_t>(0x1400)};

	inline float16_t fp16_one {static_cast<std::uint16_t>(0x3c00)};
	inline float16_t fp16_one_negative {static_cast<std::uint16_t>(0x4000)};
	inline float16_t fp16_two {static_cast<std::uint16_t>(0x4000)};
	inline float16_t fp16_two_negative {static_cast<std::uint16_t>(0xc000)};
	inline float16_t fp16_half {static_cast<std::uint16_t>(0x3800)};
	inline float16_t fp16_half_negative {static_cast<std::uint16_t>(0x3b00)};
	inline float16_t fp16_zero {static_cast<std::uint16_t>(0x0)};
	inline float16_t fp16_zero_negative {static_cast<std::uint16_t>(0x8000)};
	inline float16_t fp16_e {static_cast<std::uint16_t>(0x4170)};
	inline float16_t fp16_pi {static_cast<std::uint16_t>(0x4248)};

	inline float16_t operator+(float16_t lhs, float16_t rhs) noexcept {
		float16_t ans {lhs};
		ans += rhs;
		return ans;
	}

	inline float16_t operator-(float16_t lhs, float16_t rhs) noexcept {
		float16_t ans {lhs};
		ans -= rhs;
		return ans;
	}

	inline float16_t operator*(float16_t lhs, float16_t rhs) noexcept {
		float16_t ans {lhs};
		ans *= rhs;
		return ans;
	}

	inline float16_t operator/(float16_t lhs, float16_t rhs) noexcept {
		return float(lhs) / float(rhs);
	}

	inline bool operator<(float16_t lhs, float16_t rhs) noexcept {
		auto const &l_ieee = lhs.data_.ieee_;
		auto const &r_ieee = rhs.data_.ieee_;

		if (l_ieee.sign_ == 1) {
			if (r_ieee.sign_ == 0) return true;
			if (l_ieee.exp_ > r_ieee.exp_) return true;
			if (l_ieee.exp_ < r_ieee.exp_) return false;
			if (l_ieee.frac_ > r_ieee.frac_) return true;
			return false;
		}

		if (r_ieee.sign_ == 1) return false;
		if (l_ieee.exp_ > r_ieee.exp_) return false;
		if (l_ieee.exp_ < r_ieee.exp_) return true;
		if (l_ieee.frac_ >= r_ieee.frac_) return false;
		return true;
	}

	inline bool operator==(float16_t lhs, float16_t rhs) noexcept {
		return lhs.data_.bits_ == rhs.data_.bits_;
	}

	inline bool operator<=(float16_t lhs, float16_t rhs) noexcept {
		return (lhs < rhs) || (lhs == rhs);
	}

	inline bool operator>(float16_t lhs, float16_t rhs) noexcept { return !(lhs <= rhs); }

	inline bool operator>=(float16_t lhs, float16_t rhs) noexcept { return !(lhs < rhs); }

	inline bool operator!=(float16_t lhs, float16_t rhs) noexcept { return !(lhs == rhs); }

	template<typename CharT, class Traits>
	std::basic_ostream<CharT, Traits> &operator<<(std::basic_ostream<CharT, Traits> &os,
												  float16_t const &f) {
		// std::basic_ostringstream<CharT, Traits> __s;
		// __s.flags(os.flags());
		// __s.imbue(os.getloc());
		// __s.precision(os.precision());

		// __s << std::scientific << float(f);
		// if constexpr (float16_debug_mode) {
		// 	__s << "[0x" << std::hex << f.data_.bits_ << "]";
		// 	__s << "(";
		// 	__s << std::bitset<1>(f.data_.ieee_.sign_) << " ";
		// 	__s << std::bitset<5>(f.data_.ieee_.exp_) << " ";
		// 	__s << std::bitset<10>(f.data_.ieee_.frac_) << ")";
		// }

		// return os << __s.str();

		return os << (float)f;
	}

	template<typename CharT, class Traits>
	std::basic_istream<CharT, Traits> &operator>>(std::basic_istream<CharT, Traits> &is,
												  float16_t &f) {
		bool _fail = true;
		float _v;

		if (is >> _v) {
			_fail = false;
			f	  = _v;
		}

		if (_fail) is.setstate(std::ios_base::failbit);

		return is;
	}

	// TODO: all functions in <cmath>
	inline float16_t abs(float16_t f) noexcept {
		float16_t ans {f};
		ans.data_.bits_ &= 0x7fff;
		return ans;
	}

	namespace float16_t_private {
		auto inline make_unary_function = [](auto const &func) noexcept {
			return [func](float16_t f) -> float16_t { return (float)func(float(f)); };
		};

		auto inline make_binary_function = [](auto const &func) noexcept {
			return [func](float16_t f1, float16_t f2) -> float16_t {
				return (float)func(float(f1), float(f2));
			};
		};

		auto inline make_trinary_function = [](auto const &func) noexcept {
			return [func](float16_t f1, float16_t f2, float16_t f3) -> float16_t {
				return (float)func(float(f1), float(f2), float(f3));
			};
		};

	} // namespace float16_t_private

	inline auto fmod =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::fmod(f1, f2); });
	inline auto remainder = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::remainder(f1, f2); });
	// remquo ??
	inline auto fma = float16_t_private::make_trinary_function(
	  [](float f1, float f2, float f3) { return std::fma(f1, f2, f3); });
	inline auto fmax =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::fmax(f1, f2); });
	inline auto fmin =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::fmax(f1, f2); });
	inline auto fdim =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::fdim(f1, f2); });
	inline auto lerp = float16_t_private::make_trinary_function(
	  [](float f1, float f2, float f3) { return librapid::lerp(f1, f2, f3); });
	inline auto exp	 = float16_t_private::make_unary_function([](float f) { return std::exp(f); });
	inline auto exp2 = float16_t_private::make_unary_function([](float f) { return std::exp2(f); });
	inline auto expm1 =
	  float16_t_private::make_unary_function([](float f) { return std::expm1(f); });
	inline auto log = float16_t_private::make_unary_function([](float f) { return std::log(f); });
	inline auto log10 =
	  float16_t_private::make_unary_function([](float f) { return std::log10(f); });
	inline auto log2 = float16_t_private::make_unary_function([](float f) { return std::log2(f); });
	inline auto log1p =
	  float16_t_private::make_unary_function([](float f) { return std::log1p(f); });
	inline auto pow =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::pow(f1, f2); });
	inline auto sqrt = float16_t_private::make_unary_function([](float f) { return std::sqrt(f); });
	inline auto cbrt = float16_t_private::make_unary_function([](float f) { return std::cbrt(f); });

	inline float16_t hypot(float16_t f1, float16_t f2) { return std::hypot(float(f1), float(f2)); }

	/* -- Should be valid after c++17 for 3 parameter hypot
	inline float16_t hypot( float16_t f1, float16_t f2, float16_t )
	{
		return std::hypot( float(f1), float(f2), float(f3) );
	}
	*/

	inline auto sin	 = float16_t_private::make_unary_function([](float f) { return std::sin(f); });
	inline auto sinh = float16_t_private::make_unary_function([](float f) { return std::sinh(f); });
	inline auto cos	 = float16_t_private::make_unary_function([](float f) { return std::cos(f); });
	inline auto cosh = float16_t_private::make_unary_function([](float f) { return std::cosh(f); });
	inline auto tan	 = float16_t_private::make_unary_function([](float f) { return std::tan(f); });
	inline auto tanh = float16_t_private::make_unary_function([](float f) { return std::tanh(f); });
	inline auto asin = float16_t_private::make_unary_function([](float f) { return std::asin(f); });
	inline auto asinh =
	  float16_t_private::make_unary_function([](float f) { return std::asinh(f); });
	inline auto acos = float16_t_private::make_unary_function([](float f) { return std::acos(f); });
	inline auto acosh =
	  float16_t_private::make_unary_function([](float f) { return std::acosh(f); });
	inline auto atan = float16_t_private::make_unary_function([](float f) { return std::atan(f); });
	inline auto atanh =
	  float16_t_private::make_unary_function([](float f) { return std::atanh(f); });
	inline auto atan2 = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::atan2(f1, f2); });

	inline auto erf	 = float16_t_private::make_unary_function([](float f) { return std::erf(f); });
	inline auto erfc = float16_t_private::make_unary_function([](float f) { return std::erfc(f); });
	inline auto tgamma =
	  float16_t_private::make_unary_function([](float f) { return std::tgamma(f); });
	inline auto lgamma =
	  float16_t_private::make_unary_function([](float f) { return std::lgamma(f); });
	inline auto ceil = float16_t_private::make_unary_function([](float f) { return std::ceil(f); });
	inline auto floor =
	  float16_t_private::make_unary_function([](float f) { return std::floor(f); });
	inline auto trunc =
	  float16_t_private::make_unary_function([](float f) { return std::trunc(f); });
	inline auto round =
	  float16_t_private::make_unary_function([](float f) { return std::round(f); });
	inline auto nearbyint =
	  float16_t_private::make_unary_function([](float f) { return std::nearbyint(f); });
	inline auto rint = float16_t_private::make_unary_function([](float f) { return std::rint(f); });

	// Floating point manipulation functions not defined
	// frexp ldexp, modf, scalbn, ilogb

	inline auto logb = float16_t_private::make_unary_function([](float f) { return std::logb(f); });
	inline auto nextafter = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::nextafter(f1, f2); });
	inline auto copysign = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::copysign(f1, f2); });

	inline bool is_nan(float16_t f16) noexcept { return (std::uint16_t(f16) & 0x7fff) > 0x7f80; }

	inline bool is_inf(float16_t f16) noexcept { return (std::uint16_t(f16) & 0x7fff) == 0x7f80; }

	inline bool is_finite(float16_t f16) noexcept {
		return (std::uint16_t(f16) & 0x7f80) != 0x7f80;
	}

	inline bool is_normal(float16_t f16) noexcept {
		auto const exponent = std::uint16_t(f16) & 0x7f80;
		return (exponent != 0x7f80) && (exponent != 0);
	}

	inline bool is_positive(float16_t f16) noexcept { return ((std::uint16_t(f16)) & 0x8000) == 0; }

	inline bool is_negative(float16_t f16) noexcept { return (std::uint16_t(f16)) & 0x8000; }

	// special functions not defined
	//  assoc_laguerre, asso_legendre, hermite, legendre, laguerre, sph_bessel, sph_legendre,
	//  sph_neumann
	//

	inline auto beta =
	  float16_t_private::make_binary_function([](float f1, float f2) { return std::beta(f1, f2); });
	inline auto comp_ellint_1 =
	  float16_t_private::make_unary_function([](float f) { return std::comp_ellint_1(f); });
	inline auto comp_ellint_2 =
	  float16_t_private::make_unary_function([](float f) { return std::comp_ellint_2(f); });
	inline auto comp_ellint_3 = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::comp_ellint_3(f1, f2); });
	inline auto cyl_bessel_i = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::cyl_bessel_i(f1, f2); });
	inline auto cyl_bessel_j = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::cyl_bessel_j(f1, f2); });
	inline auto cyl_bessel_k = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::cyl_bessel_k(f1, f2); });
	inline auto cyl_neumann = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::cyl_neumann(f1, f2); });
	inline auto ellint_1 = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::ellint_1(f1, f2); });
	inline auto ellint_2 = float16_t_private::make_binary_function(
	  [](float f1, float f2) { return std::ellint_2(f1, f2); });
	inline auto ellint_3 = float16_t_private::make_trinary_function(
	  [](float f1, float f2, float f3) { return std::ellint_3(f1, f2, f3); });
	inline auto expint =
	  float16_t_private::make_unary_function([](float f) { return std::expint(f); });
	inline auto riemann_zeta =
	  float16_t_private::make_unary_function([](float f) { return std::riemann_zeta(f); });
} // namespace librapid::extended

namespace librapid::suffix {
	LR_INLINE auto operator""_h(long double val) { return extended::float16_t((float)val); }
} // namespace librapid::suffix

namespace std {
	template<>
	struct numeric_limits<librapid::extended::float16_t> {
		static constexpr bool is_specialized = true;
		static librapid::extended::float16_t min() noexcept {
			return librapid::extended::fp16_min_positive;
		}
		static librapid::extended::float16_t max() noexcept { return librapid::extended::fp16_max; }
		static librapid::extended::float16_t lowest() noexcept {
			return librapid::extended::fp16_min;
		}
		static constexpr int digits		  = 11;
		static constexpr int digits10	  = 3;
		static constexpr int max_digits10 = 4;
		static constexpr bool is_signed	  = true;
		static constexpr bool is_integer  = false;
		static constexpr bool is_exact	  = false;
		static constexpr int radix		  = std::numeric_limits<float>::radix;
		static librapid::extended::float16_t epsilon() noexcept {
			return librapid::extended::fp16_epsilon;
		}
		static librapid::extended::float16_t round_error() noexcept {
			return librapid::extended::fp16_half;
		}
		static constexpr int min_exponent					= -13;
		static constexpr int min_exponent10					= -4;
		static constexpr int max_exponent					= 14;
		static constexpr int max_exponent10					= 4;
		static constexpr bool has_infinity					= true;
		static constexpr bool has_quiet_NaN					= true;
		static constexpr bool has_signaling_NaN				= has_quiet_NaN;
		static constexpr std::float_denorm_style has_denorm = denorm_present;
		static constexpr bool has_denorm_loss				= false;
		static librapid::extended::float16_t infinity() noexcept {
			return librapid::extended::fp16_infinity;
		}
		static librapid::extended::float16_t quiet_NaN() noexcept {
			return librapid::extended::fp16_nan;
		}
		static librapid::extended::float16_t signaling_NaN() noexcept {
			return librapid::extended::fp16_nan;
		} // TODO: emit signal??
		static librapid::extended::float16_t denorm_min() noexcept {
			return librapid::extended::fp16_min_positive_subnormal;
		}
		static constexpr bool is_iec559 =
		  has_infinity && has_quiet_NaN && has_denorm == denorm_present;
		static constexpr bool is_bounded					= true;
		static constexpr bool is_modulo						= false;
		static constexpr bool traps							= false;
		static constexpr bool tinyness_before				= true;
		static constexpr std::float_round_style round_style = round_to_nearest;
	};

	template<>
	inline constexpr bool is_floating_point_v<librapid::extended::float16_t> = true;
	template<>
	inline constexpr bool is_arithmetic_v<librapid::extended::float16_t> = true;
	template<>
	inline constexpr bool is_signed_v<librapid::extended::float16_t> = true;

} // namespace std

namespace librapid {
	using half = extended::float16_t;

#define LIMIT_IMPL(NAME_) static auto NAME_() noexcept
#define NUM_LIM(NAME_)	  std::numeric_limits<Scalar>::NAME_()

	namespace internal {
		//------- 16bit Floating Point --------------------------------------------
		template<>
		struct traits<extended::float16_t> {
			static constexpr bool IsScalar		 = true;
			using Valid							 = std::true_type;
			using Scalar						 = extended::float16_t;
			using BaseScalar					 = extended::float16_t;
			using StorageType					 = memory::DenseStorage<extended::float16_t>;
			using Packet						 = std::false_type;
			using Device						 = device::CPU;
			static constexpr int64_t PacketWidth = 1;
			static constexpr char Name[]		 = "__half";
			static constexpr uint64_t Flags = flags::PacketArithmetic | flags::ScalarArithmetic |
											  flags::PacketLogical | flags::ScalarLogical;

			static constexpr uint64_t Size	= 2;
			static constexpr bool CanAlign	= true;
			static constexpr bool CanMemcpy = true;

			template<typename CAST>
			LR_FORCE_INLINE static CAST cast(const extended::float16_t &val) {
				return (CAST)val;
			}

			LR_FORCE_INLINE LIMIT_IMPL(min) { return NUM_LIM(min); }
			LR_FORCE_INLINE LIMIT_IMPL(max) { return NUM_LIM(max); }
			LR_FORCE_INLINE LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
			LR_FORCE_INLINE LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
			LR_FORCE_INLINE LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
			LR_FORCE_INLINE LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
			LR_FORCE_INLINE LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
			LR_FORCE_INLINE LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};
	} // namespace internal

#undef LIMIT_IMPL
#undef NUM_LIM
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<>
struct fmt::formatter<librapid::extended::float16_t> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::extended::float16_t &flt, FormatContext &ctx) {
		try {
			std::stringstream ss;
			ss << flt;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API
