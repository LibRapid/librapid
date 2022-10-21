set(PROJECT_NAME lib_mpir_gc)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "../../../gmp-h.in"
)
source_group("" FILES ${no_group_source_files})

set(Header_Files
    "../../../config.h"
    "../../../gmp-impl.h"
    "../../../gmp-mparam.h"
    "../../../longlong.h"
    "../../../mpir.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "../../../assert.c"
    "../../../errno.c"
    "../../../extract-dbl.c"
    "../../../invalid.c"
    "../../../memory.c"
    "../../../mp_bpl.c"
    "../../../mp_clz_tab.c"
    "../../../mp_dv_tab.c"
    "../../../mp_get_fns.c"
    "../../../mp_minv_tab.c"
    "../../../mp_set_fns.c"
    "../../../nextprime.c"
    "../../../primesieve.c"
    "../../../randbui.c"
    "../../../randclr.c"
    "../../../randdef.c"
    "../../../randiset.c"
    "../../../randlc2s.c"
    "../../../randlc2x.c"
    "../../../randmt.c"
    "../../../randmts.c"
    "../../../randmui.c"
    "../../../rands.c"
    "../../../randsd.c"
    "../../../randsdui.c"
    "../../../tal-reent.c"
    "../../../version.c"
)
source_group("Source Files" FILES ${Source_Files})

set(Source_Files__fft
    "../../../fft/adjust.c"
    "../../../fft/adjust_sqrt2.c"
    "../../../fft/butterfly_lshB.c"
    "../../../fft/butterfly_rshB.c"
    "../../../fft/combine_bits.c"
    "../../../fft/div_2expmod_2expp1.c"
    "../../../fft/fermat_to_mpz.c"
    "../../../fft/fft_mfa_trunc_sqrt2.c"
    "../../../fft/fft_mfa_trunc_sqrt2_inner.c"
    "../../../fft/fft_negacyclic.c"
    "../../../fft/fft_radix2.c"
    "../../../fft/fft_trunc.c"
    "../../../fft/fft_trunc_sqrt2.c"
    "../../../fft/ifft_mfa_trunc_sqrt2.c"
    "../../../fft/ifft_negacyclic.c"
    "../../../fft/ifft_radix2.c"
    "../../../fft/ifft_trunc.c"
    "../../../fft/ifft_trunc_sqrt2.c"
    "../../../fft/mul_2expmod_2expp1.c"
    "../../../fft/mul_fft_main.c"
    "../../../fft/mul_mfa_trunc_sqrt2.c"
    "../../../fft/mul_trunc_sqrt2.c"
    "../../../fft/mulmod_2expp1.c"
    "../../../fft/normmod_2expp1.c"
    "../../../fft/revbin.c"
    "../../../fft/split_bits.c"
)
source_group("Source Files\\fft" FILES ${Source_Files__fft})

set(Source_Files__mpf
    "../../../mpf/abs.c"
    "../../../mpf/add.c"
    "../../../mpf/add_ui.c"
    "../../../mpf/ceilfloor.c"
    "../../../mpf/clear.c"
    "../../../mpf/clears.c"
    "../../../mpf/cmp.c"
    "../../../mpf/cmp_d.c"
    "../../../mpf/cmp_si.c"
    "../../../mpf/cmp_ui.c"
    "../../../mpf/cmp_z.c"
    "../../../mpf/div.c"
    "../../../mpf/div_2exp.c"
    "../../../mpf/div_ui.c"
    "../../../mpf/dump.c"
    "../../../mpf/eq.c"
    "../../../mpf/fits_si.c"
    "../../../mpf/fits_sint.c"
    "../../../mpf/fits_slong.c"
    "../../../mpf/fits_sshort.c"
    "../../../mpf/fits_ui.c"
    "../../../mpf/fits_uint.c"
    "../../../mpf/fits_ulong.c"
    "../../../mpf/fits_ushort.c"
    "../../../mpf/get_d.c"
    "../../../mpf/get_d_2exp.c"
    "../../../mpf/get_dfl_prec.c"
    "../../../mpf/get_prc.c"
    "../../../mpf/get_si.c"
    "../../../mpf/get_str.c"
    "../../../mpf/get_ui.c"
    "../../../mpf/init.c"
    "../../../mpf/init2.c"
    "../../../mpf/inits.c"
    "../../../mpf/inp_str.c"
    "../../../mpf/int_p.c"
    "../../../mpf/iset.c"
    "../../../mpf/iset_d.c"
    "../../../mpf/iset_si.c"
    "../../../mpf/iset_str.c"
    "../../../mpf/iset_ui.c"
    "../../../mpf/mul.c"
    "../../../mpf/mul_2exp.c"
    "../../../mpf/mul_ui.c"
    "../../../mpf/neg.c"
    "../../../mpf/out_str.c"
    "../../../mpf/pow_ui.c"
    "../../../mpf/random2.c"
    "../../../mpf/reldiff.c"
    "../../../mpf/rrandomb.c"
    "../../../mpf/set.c"
    "../../../mpf/set_d.c"
    "../../../mpf/set_dfl_prec.c"
    "../../../mpf/set_prc.c"
    "../../../mpf/set_prc_raw.c"
    "../../../mpf/set_q.c"
    "../../../mpf/set_si.c"
    "../../../mpf/set_str.c"
    "../../../mpf/set_ui.c"
    "../../../mpf/set_z.c"
    "../../../mpf/size.c"
    "../../../mpf/sqrt.c"
    "../../../mpf/sqrt_ui.c"
    "../../../mpf/sub.c"
    "../../../mpf/sub_ui.c"
    "../../../mpf/swap.c"
    "../../../mpf/trunc.c"
    "../../../mpf/ui_div.c"
    "../../../mpf/ui_sub.c"
    "../../../mpf/urandomb.c"
)
source_group("Source Files\\mpf" FILES ${Source_Files__mpf})

set(Source_Files__mpn
    "../../../mpn/generic/add.c"
    "../../../mpn/generic/add_1.c"
    "../../../mpn/generic/add_err1_n.c"
    "../../../mpn/generic/add_err2_n.c"
    "../../../mpn/generic/add_n.c"
    "../../../mpn/generic/addadd_n.c"
    "../../../mpn/generic/addmul_1.c"
    "../../../mpn/generic/addsub_n.c"
    "../../../mpn/generic/and_n.c"
    "../../../mpn/generic/andn_n.c"
    "../../../mpn/generic/bdivmod.c"
    "../../../mpn/generic/binvert.c"
    "../../../mpn/generic/cmp.c"
    "../../../mpn/generic/com_n.c"
    "../../../mpn/generic/comb_tables.c"
    "../../../mpn/generic/copyd.c"
    "../../../mpn/generic/copyi.c"
    "../../../mpn/generic/dc_bdiv_q.c"
    "../../../mpn/generic/dc_bdiv_q_n.c"
    "../../../mpn/generic/dc_bdiv_qr.c"
    "../../../mpn/generic/dc_bdiv_qr_n.c"
    "../../../mpn/generic/dc_div_q.c"
    "../../../mpn/generic/dc_div_qr.c"
    "../../../mpn/generic/dc_div_qr_n.c"
    "../../../mpn/generic/dc_divappr_q.c"
    "../../../mpn/generic/divexact.c"
    "../../../mpn/generic/divexact_1.c"
    "../../../mpn/generic/divexact_by3c.c"
    "../../../mpn/generic/divexact_byff.c"
    "../../../mpn/generic/divexact_byfobm1.c"
    "../../../mpn/generic/divisible_p.c"
    "../../../mpn/generic/divrem.c"
    "../../../mpn/generic/divrem_1.c"
    "../../../mpn/generic/divrem_2.c"
    "../../../mpn/generic/divrem_euclidean_qr_1.c"
    "../../../mpn/generic/divrem_euclidean_qr_2.c"
    "../../../mpn/generic/divrem_euclidean_r_1.c"
    "../../../mpn/generic/divrem_hensel_qr_1.c"
    "../../../mpn/generic/divrem_hensel_qr_1_1.c"
    "../../../mpn/generic/divrem_hensel_qr_1_2.c"
    "../../../mpn/generic/divrem_hensel_r_1.c"
    "../../../mpn/generic/divrem_hensel_rsh_qr_1.c"
    "../../../mpn/generic/dump.c"
    "../../../mpn/generic/fib2_ui.c"
    "../../../mpn/generic/fib_table.c"
    "../../../mpn/generic/gcd.c"
    "../../../mpn/generic/gcd_1.c"
    "../../../mpn/generic/gcd_subdiv_step.c"
    "../../../mpn/generic/gcdext.c"
    "../../../mpn/generic/gcdext_1.c"
    "../../../mpn/generic/gcdext_lehmer.c"
    "../../../mpn/generic/get_d.c"
    "../../../mpn/generic/get_str.c"
    "../../../mpn/generic/hamdist.c"
    "../../../mpn/generic/hgcd.c"
    "../../../mpn/generic/hgcd2.c"
    "../../../mpn/generic/hgcd2_jacobi.c"
    "../../../mpn/generic/hgcd_appr.c"
    "../../../mpn/generic/hgcd_jacobi.c"
    "../../../mpn/generic/hgcd_matrix.c"
    "../../../mpn/generic/hgcd_reduce.c"
    "../../../mpn/generic/hgcd_step.c"
    "../../../mpn/generic/inv_div_q.c"
    "../../../mpn/generic/inv_div_qr.c"
    "../../../mpn/generic/inv_div_qr_n.c"
    "../../../mpn/generic/inv_divappr_q.c"
    "../../../mpn/generic/inv_divappr_q_n.c"
    "../../../mpn/generic/invert.c"
    "../../../mpn/generic/ior_n.c"
    "../../../mpn/generic/iorn_n.c"
    "../../../mpn/generic/jacobi.c"
    "../../../mpn/generic/jacobi_2.c"
    "../../../mpn/generic/jacobi_base.c"
    "../../../mpn/generic/lshift.c"
    "../../../mpn/generic/matrix22_mul.c"
    "../../../mpn/generic/matrix22_mul1_inverse_vector.c"
    "../../../mpn/generic/mod_1.c"
    "../../../mpn/generic/mod_1_1.c"
    "../../../mpn/generic/mod_1_2.c"
    "../../../mpn/generic/mod_1_3.c"
    "../../../mpn/generic/mod_34lsub1.c"
    "../../../mpn/generic/modexact_1c_odd.c"
    "../../../mpn/generic/mp_bases.c"
    "../../../mpn/generic/mul.c"
    "../../../mpn/generic/mul_1.c"
    "../../../mpn/generic/mul_basecase.c"
    "../../../mpn/generic/mul_fft.c"
    "../../../mpn/generic/mul_n.c"
    "../../../mpn/generic/mulhigh_n.c"
    "../../../mpn/generic/mullow_basecase.c"
    "../../../mpn/generic/mullow_n.c"
    "../../../mpn/generic/mullow_n_basecase.c"
    "../../../mpn/generic/mulmid.c"
    "../../../mpn/generic/mulmid_basecase.c"
    "../../../mpn/generic/mulmid_n.c"
    "../../../mpn/generic/mulmod_2expm1.c"
    "../../../mpn/generic/mulmod_2expp1_basecase.c"
    "../../../mpn/generic/mulmod_bexpp1.c"
    "../../../mpn/generic/nand_n.c"
    "../../../mpn/generic/neg_n.c"
    "../../../mpn/generic/nior_n.c"
    "../../../mpn/generic/nsumdiff_n.c"
    "../../../mpn/generic/perfect_square_p.c"
    "../../../mpn/generic/popcount.c"
    "../../../mpn/generic/pow_1.c"
    "../../../mpn/generic/powlo.c"
    "../../../mpn/generic/powm.c"
    "../../../mpn/generic/preinv_divrem_1.c"
    "../../../mpn/generic/preinv_mod_1.c"
    "../../../mpn/generic/random.c"
    "../../../mpn/generic/random2.c"
    "../../../mpn/generic/randomb.c"
    "../../../mpn/generic/redc_1.c"
    "../../../mpn/generic/redc_2.c"
    "../../../mpn/generic/redc_n.c"
    "../../../mpn/generic/rootrem.c"
    "../../../mpn/generic/rootrem_basecase.c"
    "../../../mpn/generic/rrandom.c"
    "../../../mpn/generic/rsh_divrem_hensel_qr_1.c"
    "../../../mpn/generic/rsh_divrem_hensel_qr_1_1.c"
    "../../../mpn/generic/rsh_divrem_hensel_qr_1_2.c"
    "../../../mpn/generic/rshift.c"
    "../../../mpn/generic/sb_bdiv_q.c"
    "../../../mpn/generic/sb_bdiv_qr.c"
    "../../../mpn/generic/sb_div_q.c"
    "../../../mpn/generic/sb_div_qr.c"
    "../../../mpn/generic/sb_divappr_q.c"
    "../../../mpn/generic/scan0.c"
    "../../../mpn/generic/scan1.c"
    "../../../mpn/generic/set_str.c"
    "../../../mpn/generic/sizeinbase.c"
    "../../../mpn/generic/sqr_basecase.c"
    "../../../mpn/generic/sqrtrem.c"
    "../../../mpn/generic/sub.c"
    "../../../mpn/generic/sub_1.c"
    "../../../mpn/generic/sub_err1_n.c"
    "../../../mpn/generic/sub_err2_n.c"
    "../../../mpn/generic/sub_n.c"
    "../../../mpn/generic/subadd_n.c"
    "../../../mpn/generic/submul_1.c"
    "../../../mpn/generic/sumdiff_n.c"
    "../../../mpn/generic/tdiv_q.c"
    "../../../mpn/generic/tdiv_qr.c"
    "../../../mpn/generic/toom3_mul.c"
    "../../../mpn/generic/toom3_mul_n.c"
    "../../../mpn/generic/toom42_mulmid.c"
    "../../../mpn/generic/toom4_mul.c"
    "../../../mpn/generic/toom4_mul_n.c"
    "../../../mpn/generic/toom8_sqr_n.c"
    "../../../mpn/generic/toom8h_mul.c"
    "../../../mpn/generic/toom_couple_handling.c"
    "../../../mpn/generic/toom_eval_dgr3_pm1.c"
    "../../../mpn/generic/toom_eval_dgr3_pm2.c"
    "../../../mpn/generic/toom_eval_pm1.c"
    "../../../mpn/generic/toom_eval_pm2.c"
    "../../../mpn/generic/toom_eval_pm2exp.c"
    "../../../mpn/generic/toom_eval_pm2rexp.c"
    "../../../mpn/generic/toom_interpolate_16pts.c"
    "../../../mpn/generic/urandomb.c"
    "../../../mpn/generic/urandomm.c"
    "../../../mpn/generic/xnor_n.c"
    "../../../mpn/generic/xor_n.c"
    "../../../mpn/generic/zero.c"
    "../../../mpn/generic/zero_p.c"
)
source_group("Source Files\\mpn" FILES ${Source_Files__mpn})

set(Source_Files__mpq
    "../../../mpq/abs.c"
    "../../../mpq/aors.c"
    "../../../mpq/canonicalize.c"
    "../../../mpq/clear.c"
    "../../../mpq/clears.c"
    "../../../mpq/cmp.c"
    "../../../mpq/cmp_si.c"
    "../../../mpq/cmp_ui.c"
    "../../../mpq/div.c"
    "../../../mpq/equal.c"
    "../../../mpq/get_d.c"
    "../../../mpq/get_den.c"
    "../../../mpq/get_num.c"
    "../../../mpq/get_str.c"
    "../../../mpq/init.c"
    "../../../mpq/inits.c"
    "../../../mpq/inp_str.c"
    "../../../mpq/inv.c"
    "../../../mpq/md_2exp.c"
    "../../../mpq/mul.c"
    "../../../mpq/neg.c"
    "../../../mpq/out_str.c"
    "../../../mpq/set.c"
    "../../../mpq/set_d.c"
    "../../../mpq/set_den.c"
    "../../../mpq/set_f.c"
    "../../../mpq/set_num.c"
    "../../../mpq/set_si.c"
    "../../../mpq/set_str.c"
    "../../../mpq/set_ui.c"
    "../../../mpq/set_z.c"
    "../../../mpq/swap.c"
)
source_group("Source Files\\mpq" FILES ${Source_Files__mpq})

set(Source_Files__mpz
    "../../../mpz/2fac_ui.c"
    "../../../mpz/abs.c"
    "../../../mpz/add.c"
    "../../../mpz/add_ui.c"
    "../../../mpz/and.c"
    "../../../mpz/aorsmul.c"
    "../../../mpz/aorsmul_i.c"
    "../../../mpz/array_init.c"
    "../../../mpz/bin_ui.c"
    "../../../mpz/bin_uiui.c"
    "../../../mpz/cdiv_q.c"
    "../../../mpz/cdiv_q_ui.c"
    "../../../mpz/cdiv_qr.c"
    "../../../mpz/cdiv_qr_ui.c"
    "../../../mpz/cdiv_r.c"
    "../../../mpz/cdiv_r_ui.c"
    "../../../mpz/cdiv_ui.c"
    "../../../mpz/cfdiv_q_2exp.c"
    "../../../mpz/cfdiv_r_2exp.c"
    "../../../mpz/clear.c"
    "../../../mpz/clears.c"
    "../../../mpz/clrbit.c"
    "../../../mpz/cmp.c"
    "../../../mpz/cmp_d.c"
    "../../../mpz/cmp_si.c"
    "../../../mpz/cmp_ui.c"
    "../../../mpz/cmpabs.c"
    "../../../mpz/cmpabs_d.c"
    "../../../mpz/cmpabs_ui.c"
    "../../../mpz/com.c"
    "../../../mpz/combit.c"
    "../../../mpz/cong.c"
    "../../../mpz/cong_2exp.c"
    "../../../mpz/cong_ui.c"
    "../../../mpz/dive_ui.c"
    "../../../mpz/divegcd.c"
    "../../../mpz/divexact.c"
    "../../../mpz/divis.c"
    "../../../mpz/divis_2exp.c"
    "../../../mpz/divis_ui.c"
    "../../../mpz/dump.c"
    "../../../mpz/export.c"
    "../../../mpz/fac_ui.c"
    "../../../mpz/fdiv_q.c"
    "../../../mpz/fdiv_q_ui.c"
    "../../../mpz/fdiv_qr.c"
    "../../../mpz/fdiv_qr_ui.c"
    "../../../mpz/fdiv_r.c"
    "../../../mpz/fdiv_r_ui.c"
    "../../../mpz/fdiv_ui.c"
    "../../../mpz/fib2_ui.c"
    "../../../mpz/fib_ui.c"
    "../../../mpz/fits_si.c"
    "../../../mpz/fits_sint.c"
    "../../../mpz/fits_slong.c"
    "../../../mpz/fits_sshort.c"
    "../../../mpz/fits_ui.c"
    "../../../mpz/fits_uint.c"
    "../../../mpz/fits_ulong.c"
    "../../../mpz/fits_ushort.c"
    "../../../mpz/gcd.c"
    "../../../mpz/gcd_ui.c"
    "../../../mpz/gcdext.c"
    "../../../mpz/get_d.c"
    "../../../mpz/get_d_2exp.c"
    "../../../mpz/get_si.c"
    "../../../mpz/get_str.c"
    "../../../mpz/get_sx.c"
    "../../../mpz/get_ui.c"
    "../../../mpz/get_ux.c"
    "../../../mpz/getlimbn.c"
    "../../../mpz/hamdist.c"
    "../../../mpz/import.c"
    "../../../mpz/init.c"
    "../../../mpz/init2.c"
    "../../../mpz/inits.c"
    "../../../mpz/inp_raw.c"
    "../../../mpz/inp_str.c"
    "../../../mpz/invert.c"
    "../../../mpz/ior.c"
    "../../../mpz/iset.c"
    "../../../mpz/iset_d.c"
    "../../../mpz/iset_si.c"
    "../../../mpz/iset_str.c"
    "../../../mpz/iset_sx.c"
    "../../../mpz/iset_ui.c"
    "../../../mpz/iset_ux.c"
    "../../../mpz/jacobi.c"
    "../../../mpz/kronsz.c"
    "../../../mpz/kronuz.c"
    "../../../mpz/kronzs.c"
    "../../../mpz/kronzu.c"
    "../../../mpz/lcm.c"
    "../../../mpz/lcm_ui.c"
    "../../../mpz/likely_prime_p.c"
    "../../../mpz/limbs_finish.c"
    "../../../mpz/limbs_modify.c"
    "../../../mpz/limbs_read.c"
    "../../../mpz/limbs_write.c"
    "../../../mpz/lucnum2_ui.c"
    "../../../mpz/lucnum_ui.c"
    "../../../mpz/mfac_uiui.c"
    "../../../mpz/miller_rabin.c"
    "../../../mpz/millerrabin.c"
    "../../../mpz/mod.c"
    "../../../mpz/mul.c"
    "../../../mpz/mul_2exp.c"
    "../../../mpz/mul_si.c"
    "../../../mpz/mul_ui.c"
    "../../../mpz/n_pow_ui.c"
    "../../../mpz/neg.c"
    "../../../mpz/next_prime_candidate.c"
    "../../../mpz/nextprime.c"
    "../../../mpz/nthroot.c"
    "../../../mpz/oddfac_1.c"
    "../../../mpz/out_raw.c"
    "../../../mpz/out_str.c"
    "../../../mpz/perfpow.c"
    "../../../mpz/perfsqr.c"
    "../../../mpz/popcount.c"
    "../../../mpz/pow_ui.c"
    "../../../mpz/powm.c"
    "../../../mpz/powm_ui.c"
    "../../../mpz/pprime_p.c"
    "../../../mpz/primorial_ui.c"
    "../../../mpz/probable_prime_p.c"
    "../../../mpz/prodlimbs.c"
    "../../../mpz/realloc.c"
    "../../../mpz/realloc2.c"
    "../../../mpz/remove.c"
    "../../../mpz/roinit_n.c"
    "../../../mpz/root.c"
    "../../../mpz/rootrem.c"
    "../../../mpz/rrandomb.c"
    "../../../mpz/scan0.c"
    "../../../mpz/scan1.c"
    "../../../mpz/set.c"
    "../../../mpz/set_d.c"
    "../../../mpz/set_f.c"
    "../../../mpz/set_q.c"
    "../../../mpz/set_si.c"
    "../../../mpz/set_str.c"
    "../../../mpz/set_sx.c"
    "../../../mpz/set_ui.c"
    "../../../mpz/set_ux.c"
    "../../../mpz/setbit.c"
    "../../../mpz/size.c"
    "../../../mpz/sizeinbase.c"
    "../../../mpz/sqrt.c"
    "../../../mpz/sqrtrem.c"
    "../../../mpz/sub.c"
    "../../../mpz/sub_ui.c"
    "../../../mpz/swap.c"
    "../../../mpz/tdiv_q.c"
    "../../../mpz/tdiv_q_2exp.c"
    "../../../mpz/tdiv_q_ui.c"
    "../../../mpz/tdiv_qr.c"
    "../../../mpz/tdiv_qr_ui.c"
    "../../../mpz/tdiv_r.c"
    "../../../mpz/tdiv_r_2exp.c"
    "../../../mpz/tdiv_r_ui.c"
    "../../../mpz/tdiv_ui.c"
    "../../../mpz/trial_division.c"
    "../../../mpz/tstbit.c"
    "../../../mpz/ui_pow_ui.c"
    "../../../mpz/ui_sub.c"
    "../../../mpz/urandomb.c"
    "../../../mpz/urandomm.c"
    "../../../mpz/xor.c"
)
source_group("Source Files\\mpz" FILES ${Source_Files__mpz})

set(Source_Files__printf
    "../../../printf/asprintf.c"
    "../../../printf/asprntffuns.c"
    "../../../printf/doprnt.c"
    "../../../printf/doprntf.c"
    "../../../printf/doprnti.c"
    "../../../printf/fprintf.c"
    "../../../printf/obprintf.c"
    "../../../printf/obprntffuns.c"
    "../../../printf/obvprintf.c"
    "../../../printf/printf.c"
    "../../../printf/printffuns.c"
    "../../../printf/repl-vsnprintf.c"
    "../../../printf/snprintf.c"
    "../../../printf/snprntffuns.c"
    "../../../printf/sprintf.c"
    "../../../printf/sprintffuns.c"
    "../../../printf/vasprintf.c"
    "../../../printf/vfprintf.c"
    "../../../printf/vprintf.c"
    "../../../printf/vsnprintf.c"
    "../../../printf/vsprintf.c"
)
source_group("Source Files\\printf" FILES ${Source_Files__printf})

set(Source_Files__scanf
    "../../../scanf/doscan.c"
    "../../../scanf/fscanf.c"
    "../../../scanf/fscanffuns.c"
    "../../../scanf/scanf.c"
    "../../../scanf/sscanf.c"
    "../../../scanf/sscanffuns.c"
    "../../../scanf/vfscanf.c"
    "../../../scanf/vscanf.c"
    "../../../scanf/vsscanf.c"
)
source_group("Source Files\\scanf" FILES ${Source_Files__scanf})

set(ALL_FILES
    ${no_group_source_files}
    ${Header_Files}
    ${Source_Files}
    ${Source_Files__fft}
    ${Source_Files__mpf}
    ${Source_Files__mpn}
    ${Source_Files__mpq}
    ${Source_Files__mpz}
    ${Source_Files__printf}
    ${Source_Files__scanf}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{MPIR_Props_External}")
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    use_props(${PROJECT_NAME} Debug   "../../mpir_debug_lib.cmake")
    use_props(${PROJECT_NAME} Release "../../mpir_release_lib.cmake")
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    use_props(${PROJECT_NAME} Debug   "../../mpir_debug_lib.cmake")
    use_props(${PROJECT_NAME} Release "../../mpir_release_lib.cmake")
endif()

set(ROOT_NAMESPACE mpir)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpir"
        TARGET_NAME_RELEASE "mpir"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpir"
        TARGET_NAME_RELEASE "mpir"
    )
endif()
################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_LIB;"
        "HAVE_CONFIG_H"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_LIB;"
        "HAVE_CONFIG_H;"
        "_WIN64"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
endif()

################################################################################
# Pre build events
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> prebuild gc Win32 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> prebuild gc Win32 22
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> prebuild gc x64 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> prebuild gc x64 22
    )
endif()

################################################################################
# Post build events
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
    )
endif()

