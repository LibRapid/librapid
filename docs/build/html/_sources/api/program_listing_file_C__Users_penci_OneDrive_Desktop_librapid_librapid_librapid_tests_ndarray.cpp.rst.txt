
.. _program_listing_file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_tests_ndarray.cpp:

Program Listing for File ndarray.cpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_tests_ndarray.cpp>` (``C:\Users\penci\OneDrive\Desktop\librapid\librapid\librapid\tests\ndarray.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #include <iostream>
   #include <chrono>
   
   #define ND_MAX_DIMS 32
   #define ND_NUM_THREADS 5
   #include <ndarray/ndarray.hpp>
   
   int main()
   {
       auto lhs = ndarray::basic_ndarray<int>(ndarray::extent({2, 2}), 0);
       auto rhs = ndarray::basic_ndarray<double>(ndarray::extent({2, 2, 1}), 0);
   
       for (nd_int i = 0; i < ndarray::math::product(lhs.get_extent().get_extent(), lhs.get_extent().ndim()); i++)
           lhs.set_value(i, i + 1);
   
       for (nd_int i = 0; i < ndarray::math::product(rhs.get_extent().get_extent(), rhs.get_extent().ndim()); i++)
           rhs.set_value(i, i + 1);
   
       std::cout << "LHS:\n" << lhs.str() << "\n\n";
       std::cout << "RHS:\n" << rhs.str() << "\n\n";
   
       try
       {
           std::cout << "Result:\n" << (10. + lhs).str() << "\n\n";
       }
       catch (std::exception &e)
       {
           std::cout << "Error occurred while adding arrays: " << e.what() << "\n";
       }
   
       return 0;
   }
