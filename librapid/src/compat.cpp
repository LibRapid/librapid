#include <librapid/librapid.hpp>

#if defined(LIBRAPID_HAS_OPENCL)

namespace clblast {
    // template<>
    // StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
    // 						   const Transpose b_transpose, const size_t m, const size_t n,
    // 						   const size_t k, const librapid::Complex<float> alpha,
    // 						   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
    // 						   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
    // 						   const librapid::Complex<float> beta, cl_mem c_buffer,
    // 						   const size_t c_offset, const size_t c_ld, cl_command_queue *queue,
    // 						   cl_event *event, cl_mem temp_buffer) {
    // 	return Gemm<float2>(layout,
    // 						a_transpose,
    // 						b_transpose,
    // 						m,
    // 						n,
    // 						k,
    // 						{alpha.real(), alpha.imag()},
    // 						a_buffer,
    // 						a_offset,
    // 						a_ld,
    // 						b_buffer,
    // 						b_offset,
    // 						b_ld,
    // 						{beta.real(), beta.imag()},
    // 						c_buffer,
    // 						c_offset,
    // 						c_ld,
    // 						queue,
    // 						event,
    // 						temp_buffer);
    // }

    // template<>
    // StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
    // 						   const Transpose b_transpose, const size_t m, const size_t n,
    // 						   const size_t k, const librapid::Complex<double> alpha,
    // 						   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
    // 						   const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
    // 						   const librapid::Complex<double> beta, cl_mem c_buffer,
    // 						   const size_t c_offset, const size_t c_ld, cl_command_queue *queue,
    // 						   cl_event *event, cl_mem temp_buffer) {
    // 	return Gemm<double2>(layout,
    // 						 a_transpose,
    // 						 b_transpose,
    // 						 m,
    // 						 n,
    // 						 k,
    // 						 {alpha.real(), alpha.imag()},
    // 						 a_buffer,
    // 						 a_offset,
    // 						 a_ld,
    // 						 b_buffer,
    // 						 b_offset,
    // 						 b_ld,
    // 						 {beta.real(), beta.imag()},
    // 						 c_buffer,
    // 						 c_offset,
    // 						 c_ld,
    // 						 queue,
    // 						 event,
    // 						 temp_buffer);
    // }

    template<>
    StatusCode PUBLIC_API Gemm(const Layout layout, const Transpose a_transpose,
                               const Transpose b_transpose, const size_t m, const size_t n,
                               const size_t k, const librapid::half alpha, const cl_mem a_buffer,
                               const size_t a_offset, const size_t a_ld, const cl_mem b_buffer,
                               const size_t b_offset, const size_t b_ld, const librapid::half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue *queue, cl_event *event, cl_mem temp_buffer) {
        return Gemm<cl_half>(layout,
                             a_transpose,
                             b_transpose,
                             m,
                             n,
                             k,
                             alpha.data().m_bits,
                             a_buffer,
                             a_offset,
                             a_ld,
                             b_buffer,
                             b_offset,
                             b_ld,
                             beta.data().m_bits,
                             c_buffer,
                             c_offset,
                             c_ld,
                             queue,
                             event,
                             temp_buffer);
    }
} // namespace clblast

#endif // LIBRAPID_HAS_OPENCL
