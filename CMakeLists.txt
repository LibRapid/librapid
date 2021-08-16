cmake_minimum_required(VERSION 3.10)
project(librapid_)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

set (CMAKE_CXX_STANDARD 14)

# Attempt to locate the required packages
find_package(CUDAToolkit)
find_package(BLAS)
find_package(OpenMP)

# Ensure PyBind11 is accessible
add_subdirectory(pybind11)

set(ENABLE_SOMETHING AUTO CACHE STRING "Enable SOMETHING support")
set_property(CACHE ENABLE_SOMETHING PROPERTY STRINGS AUTO ON OFF)

option(USE_BLAS "Attempt to use a BLAS library" ON)
option(USE_CUDA "Attempt to use CUDA" ON)
option(USE_OMP "Attempt to use OpenMP to allow multithreading" ON)

pybind11_add_module(librapid_ librapid/pybind_librapid.cpp)

add_compile_definitions(LIBRAPID_PYTHON)

# See if OpenMP should be linked against
if (${OpenMP_FOUND})
	if (${USE_OMP})
		message(STATUS "Linking against OpenMP")
		
		# Link the required library
		target_link_libraries(librapid_ PRIVATE
							  OpenMP::OpenMP_CXX
							 )
	
		# Add the compile definition so LibRapid knows it has OpenMP
		add_compile_definitions(LIBRAPID_HAS_OMP)
	else()
		message(WARNING "OpenMP was found but is not being linked against (Value <USE_OMP> is " ${USE_OMP} ")")
	endif()
endif()

# Check if BLAS was built by CI for Python Wheels.
# If so, use this instead of any other BLAS install found
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/openblas_install")
	message(STATUS "Using OpenBLAS built by CI for Python Wheels")
	set(BLAS_FOUND TRUE)
	set(USE_BLAS TRUE)

	if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
		set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/openblas_install/lib/openblas.lib")
	else()
		set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/openblas_install/lib/libopenblas.a")
	endif()
endif()

# See if BLAS should be linked against
if (${BLAS_FOUND})
	if (${USE_BLAS})
		message(STATUS "Using BLAS (" ${BLAS_LIBRARIES} ")")

		get_filename_component(filepath ${BLAS_LIBRARIES} DIRECTORY)
		
		# Copy include files
		set(inc_path "${filepath}/../include")
		message(STATUS "Checking path ${inc_path} for include files")
		FILE(GLOB_RECURSE files "${filepath}/..")
		message(STATUS "Information: ${files}")
		if (NOT (EXISTS ${inc_path}))
			message(FATAL_ERROR "Could not locate include path for BLAS")
		endif()

		set(has_cblas OFF)
		
		if (EXISTS "${inc_path}/openblas")
			FILE(GLOB_RECURSE include_files "${inc_path}/openblas/*.*")
			foreach (file IN LISTS include_files)
				message(STATUS "Found OpenBLAS include file " ${file})
				get_filename_component(filename ${file} NAME)
				configure_file(${file} ${CMAKE_SOURCE_DIR}/librapid/blas/${filename} COPYONLY)
			endforeach()
		else()
			FILE(GLOB_RECURSE include_files "${inc_path}/*.*")
			foreach (file IN LISTS include_files)
				message(STATUS "Found include file " ${file})
				get_filename_component(inc_file ${file} NAME)
				configure_file(${file} ${CMAKE_SOURCE_DIR}/librapid/blas/${inc_file} COPYONLY)
				
				if (${inc_file} STREQUAL "cblas.h")
					set(has_cblas ON)
				endif()
			endforeach()
		endif()
		
		# Copy library files
		get_filename_component(lib_name ${BLAS_LIBRARIES} NAME)
		message(STATUS "Found library file ${lib_name}")
		configure_file(${BLAS_LIBRARIES} ${CMAKE_SOURCE_DIR}/librapid/blas/${lib_name} COPYONLY)

		# Copy binary files if on Windows
		if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
			set(bin_path "${filepath}/../bin")
			if (NOT (EXISTS ${bin_path}))
				message(FATAL_ERROR "Could not locate <bin> folder for BLAS")
			endif()

			FILE(GLOB_RECURSE include_files "${bin_path}/*.dll")
			foreach (file IN LISTS include_files)
				message(STATUS "Found binary file " ${file})
				get_filename_component(filename ${file} NAME)
				configure_file(${file} ${CMAKE_SOURCE_DIR}/librapid/blas/${filename} COPYONLY)
			endforeach()
		endif()

		# Link the required library
		target_link_libraries(librapid_ PRIVATE
								"${CMAKE_SOURCE_DIR}/librapid/blas/${lib_name}"
							 )

		target_include_directories(librapid_ PRIVATE
								"${CMAKE_SOURCE_DIR}/librapid/blas"
								"${CMAKE_SOURCE_DIR}/librapid/cudahelpers"
								"${CMAKE_SOURCE_DIR}")
	
		# Add the compile definition so LibRapid knows it has BLAS
		if (${has_cblas})
			add_compile_definitions(LIBRAPID_HAS_BLAS)
		else()
			message(WARNING "Although BLAS was found, no cblas.h file was found, so BLAS support is not enabled")
		endif()
	else()
		message(WARNING "BLAS was found but is not being linked against (Value <USE_BLAS> is " ${USE_BLAS} ")")
	endif()
endif()

# Check if CUDA should be used
if (${CUDAToolkit_FOUND})
	if (${USE_CUDA})
		message(STATUS "Using CUDA ${CUDAToolkit_VERSION}")

		target_include_directories(librapid_ PRIVATE
									${CUDA_INCLUDE_DIRS}
									${CMAKE_SOURCE_DIR}
								  )

		target_link_directories(librapid_ PRIVATE
									${CUDA_LIBRARIES}
									${CUDA_CUBLAS_LIBRARIES}
							   )

		target_link_libraries(librapid_ PRIVATE
									CUDA::cudart
									CUDA::cuda_driver
									CUDA::nvrtc
									Dbghelp
							 )

		add_compile_definitions(LIBRAPID_HAS_CUDA)
	else()
		message(WARNING "CUDA was found but is not being linked against (Value <USE_CUDA> is " ${USE_CUDA} ")")
	endif()
endif()
