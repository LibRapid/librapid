# Disable some useless warnings
function(disable_warnings target warnings)
    if (MSVC)
        foreach (warning ${warnings})
            # Disables warning 'identifier'
            target_compile_options(${target} PRIVATE /wd${warning})
        endforeach ()
    else ()
        foreach (warning ${warnings})
            target_compile_options(${target} PRIVATE -Wno-${warning})
        endforeach ()
    endif ()
endfunction()

set(msvc_warnings
        4146 # unary minus operator applied to unsigned type, result still unsigned
        4127 # conditional expression is constant
        4505 # unreferenced local function has been removed
        )

set(gcc_clang_warnings
        tautological-compare # The comparison is self-evident
        type-limits # Comparison is always true due to limited range of data type
        unused-variable # Unused variable
        )

function(auto_disable_warnings target)
    if (MSVC)
        disable_warnings(${target} "${msvc_warnings}")
    else ()
        disable_warnings(${target} "${gcc_clang_warnings}")
    endif ()
endfunction()

function(disable_all_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /w)
    else()
        target_compile_options(${target} PRIVATE -w)
    endif()
endfunction()
