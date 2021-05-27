
.. _program_listing_file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_broadcast_arrays.hpp:

Program Listing for File broadcast_arrays.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_C__Users_penci_OneDrive_Desktop_librapid_librapid_librapid_ndarray_broadcast_arrays.hpp>` (``C:\Users\penci\OneDrive\Desktop\librapid\librapid\librapid\ndarray\broadcast_arrays.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef NDARRAY_BROADCAST_ARRAYS
   #define NDARRAY_BROADCAST_ARRAYS
   
   #include "utils.hpp"
   
   namespace ndarray
   {
       namespace broadcast
       {
           template<typename A, typename B>
           static int calculate_arithmetic_mode(const A *a, nd_int dims_a, const B *b, nd_int dims_b)
           {
               // Check for direct or indirect shape match
               int mode = -1; // Addition mode
   
               nd_int prodA = math::product(a, dims_a);
               nd_int prodB = math::product(b, dims_b);
   
               nd_int leading_ones_a = 0, leading_ones_b = 0;
   
               for (nd_int i = 0; i < dims_a; i++)
                   if (a[i] == 1) leading_ones_a++;
                   else break;
   
               for (nd_int i = 0; i < dims_b; i++)
                   if (b[i] == 1) leading_ones_b++;
                   else break;
   
               if (utils::check_ptr_match(a, dims_a, b, dims_b))
               {
                   // Check for exact shape match
                   mode = 0;
               }
               else if (dims_a < dims_b &&
                        prodA == prodB &&
                        utils::check_ptr_match(a, dims_a, utils::sub_vector(b, dims_b, dims_b - dims_a)))
               {
                   // Check if last dimensions of other match *this, and all other dimensions are 1
                   // E.g. [1 2] + [[[3 4]]] => [4 6]
                   mode = 0;
               }
               else if (dims_a > dims_b &&
                        prodA == prodB &&
                        utils::check_ptr_match(b, dims_b, utils::sub_vector(a, dims_a, dims_a - dims_b)))
               {
                   // Check if last dimensions of *this match other, and all other dimensions are 1
                   // E.g. [[[1 2]]] + [3 4] => [[[4 6]]]
                   mode = 0;
               }
               else if (prodB == 1)
               {
                   // Check if other is a single value array
                   // E.g. [1 2 3] + [10] => [11 12 13]
   
                   mode = 1;
               }
               else if (prodA == 1)
               {
                   // Check if this is a single value array
                   // E.g. [10] + [1 2 3] => [11 12 13]
   
                   mode = 2;
               }
               else if (utils::check_ptr_match(b, dims_b, utils::sub_vector(a, dims_a, leading_ones_a + 1)))
               {
                   // Check for "row by row" addition
                   // E.g. [[1 2]   +   [5 6]    =>   [[ 6  8]
                   //       [3 4]]                     [ 8 10]]
                   mode = 3;
               }
               else if (utils::check_ptr_match(a, dims_a, utils::sub_vector(b, dims_b, leading_ones_b + 1)))
               {
                   // Check for reverse "row by row" addition
                   // E.g. [1 2]  +   [[3 4]     =>   [[4 6]
                   //                  [5 6]]          [6 8]]
                   mode = 4;
               }
               else if (prodA == prodB &&
                        prodA == a[0] &&
                        a[0] == b[dims_b - 1])
               {
                   // Check for grid addition
                   // E.g. [[1]    +    [3 4]    =>    [[4 5]
                   //       [2]]                        [5 6]]
                   mode = 5;
               }
               else if (prodA == prodB &&
                        prodB == b[0] &&
                        a[dims_a - 1] == b[0])
               {
                   // Check for reverse grid addition
                   // E.g. [1 2]   +    [[3]     =>    [[4 5]
                   //                    [4]]           [5 6]]
                   mode = 6;
               }
               else if (a[dims_a - 1] == 1 &&
                        utils::check_ptr_match(utils::sub_vector(a, dims_a, 0, dims_b - 1),
                        utils::sub_vector(b, dims_b, 0, dims_b - 1)))
               {
                   // Check for "column by column" addition
                   // E.g. [[1]     +    [[10 11]      =>     [[11 12]
                   //       [2]]          [12 13]]             [14 15]]
                   mode = 7;
               }
               else if (b[dims_b - 1] == 1 &&
                        utils::check_ptr_match(utils::sub_vector(a, dims_a, 0, dims_b - 1),
                        utils::sub_vector(b, dims_b, 0, dims_b - 1)))
               {
                   // Check for reverse "column by column" addition
                   // E.g.  [[1 2]    +    [[5]      =>     [[ 6  7]
                   //        [3 4]]         [6]]             [ 9 10]]
                   mode = 8;
               }
   
               return mode;
           }
       }
   }
   
   #endif // NDARRAY_BROADCAST_ARRAYS
