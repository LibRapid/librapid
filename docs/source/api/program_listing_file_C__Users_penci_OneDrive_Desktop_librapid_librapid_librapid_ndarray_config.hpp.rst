
.. _program_listing_file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_config.hpp:

Program Listing for File config.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_config.hpp>` (``C:\Users\penci\OneDrive\Desktop\librapid\librapid\librapid\ndarray\config.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef NDARRAY_CONFIG
   #define NDARRAY_CONFIG
   
   #if defined(NDEBUG) || defined(NDARRAY_NDEBUG)
   #define ND_NDEBUG
   #define ND_INLINE inline
   #else
   #define ND_DEBUG
   #define ND_INLINE
   #endif // NDEBUG || NDARRAY_DEBUG
   
   #ifdef _OPENMP
   #define ND_HAS_OMP
   #endif // _OPENMP
   
   #ifdef ND_HAS_OMP
   #include <omp.h>
   #endif // NDARRAY_HAS_OMP
   
   #ifndef ND_NUM_THREADS
   #define ND_NUM_THREADS 4
   #endif
   
   #ifndef ND_MAX_DIMS
   #define ND_MAX_DIMS 50
   #endif // ND_MAX_DIMS
   
   using nd_int = unsigned long long;
   
   #endif // NDARRAY_CONFIG
