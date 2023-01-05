function(addValgrind target)
    find_program(VALGRIND_PATH valgrind)

    if (NOT VALGRIND_PATH)
        message(WARNING "[ LIBRAPID ] Valgrind not found, cannot run ${target} under valgrind")
        return()
    endif ()

    message(STATUS "[ LIBRAPID ] Adding Valgrind target")

    set(VALGRIND_OPTIONS
            "--error-exitcode=1"
            "--leak-check=full"
            "--show-leak-kinds=all"
            "--track-origins=yes"
            "--verbose"
            )

    add_custom_target(valgrind
            COMMAND ${VALGRIND_PATH} ${VALGRIND_OPTIONS}
            $<TARGET_FILE:${target}>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
endfunction()
