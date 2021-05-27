
.. _program_listing_file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_errors.hpp:

Program Listing for File errors.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_errors.hpp>` (``C:\Users\penci\OneDrive\Desktop\librapid\librapid\librapid\ndarray\errors.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef NDARRAY_ERRORS
   #define NDARRAY_ERRORS
   
   namespace ndarray
   {
       enum class errors
       {
           ALL_OK = 0,
           INDEX_OUT_OF_RANGE = 1,
           ARRAY_DIMENSIONS_TOO_LARGE = 2
       };
   }
   
   #endif // NDARRAY_ERRORS
