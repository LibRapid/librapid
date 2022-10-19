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

void mpir_fft_mfa_trunc_sqrt2_inner(mp_ptr * ii, mp_ptr * jj, mp_size_t n, 
                   mp_bitcnt_t w, mp_ptr * t1, mp_ptr * t2, 
                  mp_ptr * temp, mp_size_t n1, mp_size_t trunc, mp_ptr tt)
{
   mp_size_t i, j, s;
   mp_size_t n2 = (2*n)/n1;
   mp_size_t trunc2 = (trunc - 2*n)/n1;
   mp_size_t limbs = (n*w)/GMP_LIMB_BITS;
   mp_bitcnt_t depth = 0;
   mp_bitcnt_t depth2 = 0;
   
   while ((((mp_size_t)1)<<depth) < n2) depth++;
   while ((((mp_size_t)1)<<depth2) < n1) depth2++;

   ii += 2*n;
   jj += 2*n;

   /* convolutions on relevant rows */
   for (s = 0; s < trunc2; s++)
   {
      i = mpir_revbin(s, depth);
      mpir_fft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
      if (ii != jj) mpir_fft_radix2(jj + i*n1, n1/2, w*n2, t1, t2);
      
      for (j = 0; j < n1; j++)
      {
         mp_size_t t = i*n1 + j;
         mpn_normmod_2expp1(ii[t], limbs);
         if (ii != jj) mpn_normmod_2expp1(jj[t], limbs);
         mpn_mulmod_Bexpp1(ii[t], ii[t], jj[t], n * w / GMP_LIMB_BITS, tt);
      }      
      
      mpir_ifft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
   }

   ii -= 2*n;
   jj -= 2*n;

   /* convolutions on rows */
   for (i = 0; i < n2; i++)
   {
      mpir_fft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
      if (ii != jj) mpir_fft_radix2(jj + i*n1, n1/2, w*n2, t1, t2);

      for (j = 0; j < n1; j++)
      {
         mp_size_t t = i*n1 + j;
         mpn_normmod_2expp1(ii[t], limbs);
         if (ii != jj) mpn_normmod_2expp1(jj[t], limbs);
         mpn_mulmod_Bexpp1(ii[t], ii[t], jj[t], n * w / GMP_LIMB_BITS, tt);
      }      
      
      mpir_ifft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
   }
}

