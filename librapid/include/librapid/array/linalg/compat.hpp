#ifndef LIBRAPID_LINALG_COMPAT_HPP
#define LIBRAPID_LINALG_COMPAT_HPP

#if defined(LIBRAPID_HAS_OPENCL)

// In an ideal world, we would be using CLBlast's datatypes directly. However, OpenCL will not
// always be available and it makes it very difficult to integrate with the rest of LibRapid.
// Instead, we will use our own datatypes and then map a few specific function calls to CLBlast
// types.

namespace clblast {
	// template<>
	// StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
	// 						   const Transpose b_transpose, const size_t m, const size_t n,
	// 						   const size_t k, const librapid::Complex<float> alpha,
	// 						   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
	// 						   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
	// 						   const librapid::Complex<float> beta, cl_mem c_buffer,
	// 						   const size_t c_offset, const size_t c_ld, cl_command_queue *queue,
	// 						   cl_event *event, cl_mem temp_buffer);

	// template<>
	// StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
	// 						   const Transpose b_transpose, const size_t m, const size_t n,
	// 						   const size_t k, const librapid::Complex<double> alpha,
	// 						   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
	// 						   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
	// 						   const librapid::Complex<double> beta, cl_mem c_buffer,
	// 						   const size_t c_offset, const size_t c_ld, cl_command_queue *queue,
	// 						   cl_event *event, cl_mem temp_buffer);

	template<>
	StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
							   const Transpose b_transpose, const size_t m, const size_t n,
							   const size_t k, const librapid::half alpha, const cl_mem a_buffer,
							   const size_t a_offset, const size_t a_ld, const cl_mem b_buffer,
							   const size_t b_offset, const size_t b_ld, const librapid::half beta,
							   cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
							   cl_command_queue *queue, cl_event *event, cl_mem temp_buffer);
} // namespace clblast

#endif // LIBRAPID_HAS_OPENCL

#endif // LIBRAPID_LINALG_COMPAT_HPP
