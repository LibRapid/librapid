
.. _program_listing_file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_array_arithmetic.hpp:

Program Listing for File array_arithmetic.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_array_arithmetic.hpp>` (``C:\Users\penci\OneDrive\Desktop\librapid\librapid\librapid\ndarray\array_arithmetic.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef NDARRAY_ARRAY_ARITHMETIC
   #define NDARRAY_ARRAY_ARITHMETIC
   
   #include <cstring> // For memset
   #include <omp.h>
   
   namespace ndarray
   {
       namespace arithmetic
       {
           template<typename A, typename B,
               typename E,
               typename S_a, typename S_b,
               typename LAMBDA>
               ND_INLINE void array_op(A *__restrict src_a, B *__restrict src_b,
                                       const basic_extent<E> &extent,
                                       const basic_stride<S_a> &stride_a,
                                       const basic_stride<S_b> &stride_b,
                                       LAMBDA op)
           {
               nd_int idim = 0;
               nd_int ndim = extent.ndim();
   
               const auto *__restrict _extent = extent.get_extent();
               const auto *__restrict _stride_a = stride_a.get_stride();
               const auto *__restrict _stride_b = stride_b.get_stride();
   
               // All strides are non-trivial
               nd_int mode = 1;
   
               // All strides trivial
               if (stride_a.is_trivial() && stride_b.is_trivial())
                   mode = 0;
   
               const auto end = math::product(extent.get_extent(), extent.ndim());
   
               nd_int coord[ND_MAX_DIMS]{};
   
               switch (mode)
               {
                   case 0:
                       {
                           if (end > 100000)
                           {
                           #pragma omp parallel for shared(src_a, src_b, op) default(none) num_threads(ND_NUM_THREADS)
                               for (long long i = 0; i < end; ++i)
                                   src_a[i] = op(src_b[i]);
                           }
                           else
                           {
                               for (long long i = 0; i < end; ++i)
                                   src_a[i] = op(src_b[i]);
                           }
                           break;
                       }
                   case 1:
                       {
                           do
                           {
                               *src_a = op(*src_b);
   
                               for (idim = 0; idim < ndim; ++idim)
                               {
                                   if (++coord[idim] == _extent[idim])
                                   {
                                       coord[idim] = 0;
                                       src_a = src_a - (_extent[idim] - 1) * _stride_a[idim];
                                       src_b = src_b - (_extent[idim] - 1) * _stride_b[idim];
                                   }
                                   else
                                   {
                                       src_a = src_a + _stride_a[idim];
                                       src_b = src_b + _stride_b[idim];
                                       break;
                                   }
                               }
                           } while (idim < ndim);
                           break;
                       }
               }
           }
       }
   }
   
   #endif // NDARRAY_ARRAY_ARITHMETIC
