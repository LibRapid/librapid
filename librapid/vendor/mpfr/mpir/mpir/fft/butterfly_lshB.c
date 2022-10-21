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

void mpir_butterfly_lshB(mp_ptr t, mp_ptr u, mp_ptr i1, 
                       mp_ptr i2, mp_size_t limbs, mp_size_t x, mp_size_t y)
{
   mp_limb_t cy, cy1;
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
   const mp_limb_t cy2 = 0;
#else
   mp_limb_t cy2;
#endif

   if (x == 0)
   {
      if (y == 0)
         cy = mpn_sumdiff_n(t + x, u + y, i1, i2, limbs + 1);
      else
      {
         cy = mpn_sumdiff_n(t, u + y, i1, i2, limbs - y);
         u[limbs] = -(cy&1);
         cy1 = cy>>1;
         cy = mpn_sumdiff_n(t + limbs - y, u, i2 + limbs - y, i1 + limbs - y, y);
         t[limbs] = cy>>1;
         mpn_add_1(t + limbs - y, t + limbs - y, y + 1, cy1);
         cy1 = -(cy&1) + (i2[limbs] - i1[limbs]);
         mpn_addmod_2expp1_1(u + y, limbs - y, cy1);
         cy1 = -(i1[limbs] + i2[limbs]);
         mpn_addmod_2expp1_1(t, limbs, cy1);
      }
   } else if (y == 0)
   {
      cy = mpn_sumdiff_n(t + x, u, i1, i2, limbs - x);
      t[limbs] = cy>>1;
      cy1 = cy&1;
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
      cy = mpn_nsumdiff_n(t, u + limbs - x, i1 + limbs - x, i2 + limbs - x, x);
#else
      cy = mpn_sumdiff_n(t, u + limbs - x, i1 + limbs - x, i2 + limbs - x, x);
      cy2 = mpn_neg_n(t, t, x);
#endif
      u[limbs] = -(cy&1);
      mpn_sub_1(u + limbs - x, u + limbs - x, x + 1, cy1);
      cy1 = -(cy>>1) - cy2;
      cy1 -= (i1[limbs] + i2[limbs]);
      mpn_addmod_2expp1_1(t + x, limbs - x, cy1);
      cy1 = (i2[limbs] - i1[limbs]);
      mpn_addmod_2expp1_1(u, limbs, cy1);
   } else if (x > y)
   {
      cy = mpn_sumdiff_n(t + x, u + y, i1, i2, limbs - x);
      t[limbs] = cy>>1;
      cy1 = cy&1;
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
      cy = mpn_nsumdiff_n(t, u + y + limbs - x, i1 + limbs - x, i2 + limbs - x, x - y);
#else
      cy = mpn_sumdiff_n(t, u + y + limbs - x, i1 + limbs - x, i2 + limbs - x, x - y);
      cy2 = mpn_neg_n(t, t, x - y);
#endif
      u[limbs] = -(cy&1);
      mpn_sub_1(u + y + limbs - x, u + y + limbs - x, x - y + 1, cy1);
      cy1 = (cy>>1) + cy2;
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
      cy = mpn_nsumdiff_n(t + x - y, u, i2 + limbs - y, i1 + limbs - y, y);
#else
      cy = mpn_sumdiff_n(t + x - y, u, i2 + limbs - y, i1 + limbs - y, y);
      cy2 = mpn_neg_n(t + x - y, t + x - y, y);
#endif
      cy1 = -(cy>>1) - mpn_sub_1(t + x - y, t + x - y, y, cy1) - cy2;
      cy1 -= (i1[limbs] + i2[limbs]);
      mpn_addmod_2expp1_1(t + x, limbs - x, cy1);
      cy1 = -(cy&1) + (i2[limbs] - i1[limbs]);
      mpn_addmod_2expp1_1(u + y, limbs - y, cy1);
   } else if (x < y)
   {
      cy = mpn_sumdiff_n(t + x, u + y, i1, i2, limbs - y);
      u[limbs] = -(cy&1);
      cy1 = cy>>1;
      cy = mpn_sumdiff_n(t + x + limbs - y, u, i2 + limbs - y, i1 + limbs - y, y - x);
      t[limbs] = cy>>1;
      mpn_add_1(t + x + limbs - y, t + x + limbs - y, y - x + 1, cy1);
      cy1 = cy&1;
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
      cy = mpn_nsumdiff_n(t, u + y - x, i2 + limbs - x, i1 + limbs - x, x);
#else
      cy = mpn_sumdiff_n(t, u + y - x, i2 + limbs - x, i1 + limbs - x, x);
#endif
      cy1 = -(cy&1) - mpn_sub_1(u + y - x, u + y - x, x, cy1);
      cy1 += (i2[limbs] - i1[limbs]);
      mpn_addmod_2expp1_1(u + y, limbs - y, cy1);
#ifndef HAVE_NATIVE_mpn_nsumdiff_n
      cy2 = mpn_neg_n(t, t, x);
#endif
      cy1 = -(cy>>1) - (i1[limbs] + i2[limbs]) - cy2;
      mpn_addmod_2expp1_1(t + x, limbs - x, cy1);
   } else /* x == y */
   {
      cy = mpn_sumdiff_n(t + x, u + x, i1, i2, limbs - x);
      t[limbs] = cy>>1;
      u[limbs] = -(cy&1);
#ifdef HAVE_NATIVE_mpn_nsumdiff_n
      cy = mpn_nsumdiff_n(t, u, i2 + limbs - x, i1 + limbs - x, x);
#else
      cy = mpn_sumdiff_n(t, u, i2 + limbs - x, i1 + limbs - x, x);
      cy2 = mpn_neg_n(t, t, x);
#endif
      cy1 = -(cy>>1) - (i1[limbs] + i2[limbs]) - cy2;
      mpn_addmod_2expp1_1(t + x, limbs - x, cy1);
      cy1 = -(cy&1) + i2[limbs] - i1[limbs];
      mpn_addmod_2expp1_1(u + x, limbs - x, cy1);
  }
}

