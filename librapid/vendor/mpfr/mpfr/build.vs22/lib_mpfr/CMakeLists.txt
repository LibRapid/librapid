set(PROJECT_NAME lib_mpfr)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "../../../mpir/lib/$(IntDir)config.h"
    "../../../mpir/lib/$(IntDir)gmp-impl.h"
    "../../../mpir/lib/$(IntDir)gmp-mparam.h"
    "../../../mpir/lib/$(IntDir)gmp.h"
    "../../../mpir/lib/$(IntDir)longlong.h"
    "../../src/fits_s.h"
    "../../src/fits_u.h"
    "../../src/gen_inverse.h"
    "../../src/ieee_floats.h"
    "../../src/mpf2mpfr.h"
    "../../src/mpfr-cvers.h"
    "../../src/mpfr-gmp.h"
    "../../src/mpfr-impl.h"
    "../../src/mpfr-intmax.h"
    "../../src/mpfr-longlong.h"
    "../../src/mpfr-mini-gmp.h"
    "../../src/mpfr-sassert.h"
    "../../src/mpfr-thread.h"
    "../../src/mpfr.h"
    "../../src/random_deviate.h"
    "../mparam.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "../../src/abort_prec_max.c"
    "../../src/acos.c"
    "../../src/acosh.c"
    "../../src/acosu.c"
    "../../src/add.c"
    "../../src/add1.c"
    "../../src/add1sp.c"
    "../../src/add_d.c"
    "../../src/add_ui.c"
    "../../src/agm.c"
    "../../src/ai.c"
    "../../src/asin.c"
    "../../src/asinh.c"
    "../../src/asinu.c"
    "../../src/atan.c"
    "../../src/atan2.c"
    "../../src/atan2u.c"
    "../../src/atanh.c"
    "../../src/atanu.c"
    "../../src/bernoulli.c"
    "../../src/beta.c"
    "../../src/buildopt.c"
    "../../src/cache.c"
    "../../src/cbrt.c"
    "../../src/check.c"
    "../../src/clear.c"
    "../../src/clears.c"
    "../../src/cmp.c"
    "../../src/cmp2.c"
    "../../src/cmp_d.c"
    "../../src/cmp_ld.c"
    "../../src/cmp_si.c"
    "../../src/cmp_ui.c"
    "../../src/cmpabs.c"
    "../../src/cmpabs_ui.c"
    "../../src/comparisons.c"
    "../../src/compound.c"
    "../../src/const_catalan.c"
    "../../src/const_euler.c"
    "../../src/const_log2.c"
    "../../src/const_pi.c"
    "../../src/constant.c"
    "../../src/copysign.c"
    "../../src/cos.c"
    "../../src/cosh.c"
    "../../src/cosu.c"
    "../../src/cot.c"
    "../../src/coth.c"
    "../../src/csc.c"
    "../../src/csch.c"
    "../../src/d_div.c"
    "../../src/d_sub.c"
    "../../src/digamma.c"
    "../../src/dim.c"
    "../../src/div.c"
    "../../src/div_2exp.c"
    "../../src/div_2si.c"
    "../../src/div_2ui.c"
    "../../src/div_d.c"
    "../../src/div_ui.c"
    "../../src/dot.c"
    "../../src/dump.c"
    "../../src/eint.c"
    "../../src/eq.c"
    "../../src/erandom.c"
    "../../src/erf.c"
    "../../src/erfc.c"
    "../../src/exceptions.c"
    "../../src/exp.c"
    "../../src/exp10.c"
    "../../src/exp10m1.c"
    "../../src/exp2.c"
    "../../src/exp2m1.c"
    "../../src/exp3.c"
    "../../src/exp_2.c"
    "../../src/expm1.c"
    "../../src/extract.c"
    "../../src/factorial.c"
    "../../src/fits_intmax.c"
    "../../src/fits_sint.c"
    "../../src/fits_slong.c"
    "../../src/fits_sshort.c"
    "../../src/fits_uint.c"
    "../../src/fits_uintmax.c"
    "../../src/fits_ulong.c"
    "../../src/fits_ushort.c"
    "../../src/fma.c"
    "../../src/fmma.c"
    "../../src/fmod_ui.c"
    "../../src/fms.c"
    "../../src/fpif.c"
    "../../src/frac.c"
    "../../src/free_cache.c"
    "../../src/frexp.c"
    "../../src/gamma.c"
    "../../src/gamma_inc.c"
    "../../src/gammaonethird.c"
    "../../src/get_d.c"
    "../../src/get_d128.c"
    "../../src/get_d64.c"
    "../../src/get_exp.c"
    "../../src/get_f.c"
    "../../src/get_float128.c"
    "../../src/get_flt.c"
    "../../src/get_ld.c"
    "../../src/get_q.c"
    "../../src/get_si.c"
    "../../src/get_sj.c"
    "../../src/get_str.c"
    "../../src/get_ui.c"
    "../../src/get_uj.c"
    "../../src/get_z.c"
    "../../src/get_z_2exp.c"
    "../../src/gmp_op.c"
    "../../src/grandom.c"
    "../../src/hypot.c"
    "../../src/init.c"
    "../../src/init2.c"
    "../../src/inits.c"
    "../../src/inits2.c"
    "../../src/inp_str.c"
    "../../src/int_ceil_log2.c"
    "../../src/isinf.c"
    "../../src/isinteger.c"
    "../../src/isnan.c"
    "../../src/isnum.c"
    "../../src/isqrt.c"
    "../../src/isregular.c"
    "../../src/iszero.c"
    "../../src/jn.c"
#    "../../src/jyn_asympt.c"
    "../../src/li2.c"
    "../../src/lngamma.c"
    "../../src/log.c"
    "../../src/log10.c"
    "../../src/log10p1.c"
    "../../src/log1p.c"
    "../../src/log2.c"
    "../../src/log2p1.c"
    "../../src/log_ui.c"
    "../../src/logging.c"
    "../../src/min_prec.c"
    "../../src/minmax.c"
    "../../src/modf.c"
#    "../../src/mp_clz_tab.c"
    "../../src/mpfr-gmp.c"
    "../../src/mpfr-mini-gmp.c"
    "../../src/mpn_exp.c"
    "../../src/mul.c"
    "../../src/mul_2exp.c"
    "../../src/mul_2si.c"
    "../../src/mul_2ui.c"
    "../../src/mul_d.c"
    "../../src/mul_ui.c"
    "../../src/mulders.c"
    "../../src/nbits_ulong.c"
    "../../src/neg.c"
    "../../src/next.c"
    "../../src/nrandom.c"
    "../../src/odd_p.c"
    "../../src/out_str.c"
    "../../src/pool.c"
    "../../src/pow.c"
    "../../src/pow_si.c"
    "../../src/pow_sj.c"
    "../../src/pow_ui.c"
    "../../src/pow_uj.c"
    "../../src/pow_z.c"
    "../../src/powerof2.c"
    "../../src/powr.c"
    "../../src/print_raw.c"
    "../../src/print_rnd_mode.c"
    "../../src/printf.c"
    "../../src/random_deviate.c"
    "../../src/rec_sqrt.c"
    "../../src/reldiff.c"
    "../../src/rem1.c"
    "../../src/rint.c"
    "../../src/rndna.c"
    "../../src/root.c"
    "../../src/round_near_x.c"
    "../../src/round_p.c"
    "../../src/round_prec.c"
#    "../../src/round_raw_generic.c"
    "../../src/scale2.c"
    "../../src/sec.c"
    "../../src/sech.c"
    "../../src/set.c"
    "../../src/set_d.c"
    "../../src/set_d128.c"
    "../../src/set_d64.c"
    "../../src/set_dfl_prec.c"
    "../../src/set_exp.c"
    "../../src/set_f.c"
    "../../src/set_float128.c"
    "../../src/set_flt.c"
    "../../src/set_inf.c"
    "../../src/set_ld.c"
    "../../src/set_nan.c"
    "../../src/set_prc_raw.c"
    "../../src/set_prec.c"
    "../../src/set_q.c"
    "../../src/set_rnd.c"
    "../../src/set_si.c"
    "../../src/set_si_2exp.c"
    "../../src/set_sj.c"
    "../../src/set_str.c"
    "../../src/set_str_raw.c"
    "../../src/set_ui.c"
    "../../src/set_ui_2exp.c"
    "../../src/set_uj.c"
    "../../src/set_z.c"
    "../../src/set_z_2exp.c"
    "../../src/set_zero.c"
    "../../src/setmax.c"
    "../../src/setmin.c"
    "../../src/setsign.c"
    "../../src/sgn.c"
    "../../src/si_op.c"
    "../../src/signbit.c"
    "../../src/sin.c"
    "../../src/sin_cos.c"
    "../../src/sinh.c"
    "../../src/sinh_cosh.c"
    "../../src/sinu.c"
    "../../src/sqr.c"
    "../../src/sqrt.c"
    "../../src/sqrt_ui.c"
    "../../src/stack_interface.c"
    "../../src/strtofr.c"
    "../../src/sub.c"
    "../../src/sub1.c"
    "../../src/sub1sp.c"
    "../../src/sub_d.c"
    "../../src/sub_ui.c"
    "../../src/subnormal.c"
    "../../src/sum.c"
    "../../src/swap.c"
    "../../src/tan.c"
    "../../src/tanh.c"
    "../../src/tanu.c"
    "../../src/total_order.c"
    "../../src/ubf.c"
    "../../src/uceil_exp2.c"
    "../../src/uceil_log2.c"
    "../../src/ufloor_log2.c"
    "../../src/ui_div.c"
    "../../src/ui_pow.c"
    "../../src/ui_pow_ui.c"
    "../../src/ui_sub.c"
    "../../src/urandom.c"
    "../../src/urandomb.c"
    "../../src/vasprintf.c"
    "../../src/version.c"
    "../../src/volatile.c"
    "../../src/yn.c"
    "../../src/zeta.c"
    "../../src/zeta_ui.c"
    "../get_patches.c"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE lib_mpfr)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpfr"
        TARGET_NAME_RELEASE "mpfr"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpfr"
        TARGET_NAME_RELEASE "mpfr"
    )
endif()
################################################################################
# Output directory
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/../lib/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../lib/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/../lib/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../lib/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/"
    )
endif()
################################################################################
# MSVC runtime library
################################################################################
get_property(MSVC_RUNTIME_LIBRARY_DEFAULT TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY ${MSVC_RUNTIME_LIBRARY_STR})

################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/..;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../mpir/lib/${CMAKE_CURRENT_BINARY_DIR}\${CMAKE_CFG_INTDIR}\"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/..;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../mpir/lib/${CMAKE_CURRENT_BINARY_DIR}\${CMAKE_CFG_INTDIR}\"
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
        "_WIN32;"
        "HAVE_CONFIG_H;"
        "_LIB;"
        "_GMP_IEEE_FLOATS;"
        "_CRT_SECURE_NO_WARNINGS;"
        "MPFR_HAVE_GMP_IMPL"
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
        "_WIN32;"
        "HAVE_CONFIG_H;"
        "_LIB;"
        "_GMP_IEEE_FLOATS;"
        "_CRT_SECURE_NO_WARNINGS;"
        "MPFR_HAVE_GMP_IMPL"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1
            >
            $<$<CONFIG:Release>:
                /Ox;
                /GS-
            >
            /Oi;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1;
                /Y-
            >
            $<$<CONFIG:Release>:
                /Ox;
                /GS-
            >
            /Oi;
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
        COMMAND   $<CONFIG:Debug> ..\\out_copy_rename.bat ..\\..\\src\\mpfr.h ..\\..\\lib\\${CMAKE_CURRENT_BINARY_DIR}\\${CMAKE_CFG_INTDIR}\\ mpfr.h
        COMMAND   $<CONFIG:Debug> ..\\out_copy_rename.bat ..\\..\\src\\mparam_h.in ..\\..\\ mparam.h
        COMMAND $<CONFIG:Release> ..\\out_copy_rename.bat ..\\..\\src\\mpfr.h ..\\..\\lib\\${CMAKE_CURRENT_BINARY_DIR}\\${CMAKE_CFG_INTDIR}\\ mpfr.h
        COMMAND $<CONFIG:Release> ..\\out_copy_rename.bat ..\\..\\src\\mparam_h.in ..\\..\\ mparam.h
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> ..\\out_copy_rename.bat ..\\..\\src\\mpfr.h ..\\..\\lib\\${CMAKE_CURRENT_BINARY_DIR}\\${CMAKE_CFG_INTDIR}\\ mpfr.h
        COMMAND   $<CONFIG:Debug> ..\\out_copy_rename.bat ..\\..\\src\\mparam_h.in ..\\..\\ mparam.h
        COMMAND $<CONFIG:Release> ..\\out_copy_rename.bat ..\\..\\src\\mpfr.h ..\\..\\lib\\${CMAKE_CURRENT_BINARY_DIR}\\${CMAKE_CFG_INTDIR}\\ mpfr.h
        COMMAND $<CONFIG:Release> ..\\out_copy_rename.bat ..\\..\\src\\mparam_h.in ..\\..\\ mparam.h
    )
endif()

################################################################################
# Dependencies
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "..\..\..\mpir\lib\$(IntDir)mpir"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "..\..\..\mpir\lib\$(IntDir)mpir"
    )
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

