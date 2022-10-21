/* 

Copyright 2009, 2011 William Hart. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY William Hart ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL William Hart OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of William Hart.

*/

#include "mpir.h"
#include "gmp-impl.h"
      
void 
mpn_mul_trunc_sqrt2(mp_ptr r1, mp_srcptr i1, mp_size_t n1, 
                        mp_srcptr i2, mp_size_t n2, mp_bitcnt_t depth, mp_bitcnt_t w)
{
   mp_size_t n = (((mp_size_t)1)<<depth);
   mp_bitcnt_t bits1 = (n*w - (depth+1))/2; 
   
   mp_size_t r_limbs = n1 + n2;
   mp_size_t limbs = (n*w)/GMP_LIMB_BITS;
   mp_size_t size = limbs + 1;

   mp_size_t j1 = (n1*GMP_LIMB_BITS - 1)/bits1 + 1;
   mp_size_t j2 = (n2*GMP_LIMB_BITS - 1)/bits1 + 1;
   
   mp_size_t i, j, trunc;

   mp_limb_t ** ii, ** jj, * t1, * t2, * s1, * tt, * ptr;
   mp_limb_t c;
   TMP_DECL;

   TMP_MARK;
   ii = TMP_BALLOC_MP_PTRS(4*(n + n*size) + 5*size);
   for (i = 0, ptr = (mp_ptr) ii + 4*n; i < 4*n; i++, ptr += size) 
   {
      ii[i] = ptr;
   }
   t1 = ptr;
   t2 = t1 + size;
   s1 = t2 + size;
   tt = s1 + size;

   if (i1 != i2)
   {
      jj = TMP_BALLOC_MP_PTRS(4*(n + n*size));
      for (i = 0, ptr = (mp_ptr) jj + 4*n; i < 4*n; i++, ptr += size) 
      {
         jj[i] = ptr;
      }
   } 
   else
      jj = ii;

   trunc = j1 + j2 - 1;
   if (trunc <= 2*n) trunc = 2*n + 1; /* trunc must be greater than 2n */
   trunc = 2*((trunc + 1)/2); /* trunc must be divisible by 2 */

   j1 = mpir_fft_split_bits(ii, i1, n1, bits1, limbs);
   for (j = j1 ; j < 4*n; j++)
      mpn_zero(ii[j], limbs + 1);
   
   mpir_fft_trunc_sqrt2(ii, n, w, &t1, &t2, &s1, trunc);
    
   if (i1 != i2)
   {
      j2 = mpir_fft_split_bits(jj, i2, n2, bits1, limbs);
      for (j = j2 ; j < 4*n; j++)
         mpn_zero(jj[j], limbs + 1);
      mpir_fft_trunc_sqrt2(jj, n, w, &t1, &t2, &s1, trunc);      
   } 
   else 
       j2 = j1;

   for (j = 0; j < trunc; j++)
   {
      mpn_normmod_2expp1(ii[j], limbs);
      if (i1 != i2) mpn_normmod_2expp1(jj[j], limbs);
      c = 2*ii[j][limbs] + jj[j][limbs];

      ii[j][limbs] = mpn_mulmod_2expp1_basecase(ii[j], ii[j], jj[j], c, n*w, tt);
   }

   mpir_ifft_trunc_sqrt2(ii, n, w, &t1, &t2, &s1, trunc);
   for (j = 0; j < trunc; j++)
   {
      mpn_div_2expmod_2expp1(ii[j], ii[j], limbs, depth + 2);
      mpn_normmod_2expp1(ii[j], limbs);
   }
   
   mpn_zero(r1, r_limbs);
   mpir_fft_combine_bits(r1, ii, j1 + j2 - 1, bits1, limbs, r_limbs);
     
   TMP_FREE;
}
