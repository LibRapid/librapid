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
      
void mpir_fft_butterfly_twiddle(mp_ptr u, mp_ptr v, 
    mp_ptr s, mp_ptr t, mp_size_t limbs, mp_bitcnt_t b1, mp_bitcnt_t b2)
{
   mp_limb_t nw = limbs*GMP_LIMB_BITS;
   mp_size_t x, y;
   int negate1 = 0;
   int negate2 = 0;
   
   if (b1 >= nw) 
   {
      negate2 = 1;
      b1 -= nw;
   }
   x  = b1/GMP_LIMB_BITS;
   b1 = b1%GMP_LIMB_BITS;

   if (b2 >= nw) 
   {
      negate1 = 1;
      b2 -= nw;
   }
   y  = b2/GMP_LIMB_BITS;
   b2 = b2%GMP_LIMB_BITS;
 
   mpir_butterfly_lshB(u, v, s, t, limbs, x, y);
   mpn_mul_2expmod_2expp1(u, u, limbs, b1);
   if (negate2) mpn_neg_n(u, u, limbs + 1);
   mpn_mul_2expmod_2expp1(v, v, limbs, b2);
   if (negate1) mpn_neg_n(v, v, limbs + 1);
}

void mpir_fft_radix2_twiddle(mp_ptr * ii, mp_size_t is,
      mp_size_t n, mp_bitcnt_t w, mp_ptr * t1, mp_ptr * t2,
      mp_size_t ws, mp_size_t r, mp_size_t c, mp_size_t rs)
{
   mp_size_t i;
   mp_size_t limbs;

#if 0
start:
#endif

   limbs = (w*n)/GMP_LIMB_BITS;
   
   if (n == 1) 
   {
      mp_size_t tw1 = r*c;
      mp_size_t tw2 = tw1 + rs*c;

      mpir_fft_butterfly_twiddle(*t1, *t2, ii[0], ii[is], limbs, tw1*ws, tw2*ws);
      MP_PTR_SWAP(ii[0],  *t1);
      MP_PTR_SWAP(ii[is], *t2);

      return;
   }

   for (i = 0; i < n; i++) 
   {   
      mpir_fft_butterfly(*t1, *t2, ii[i*is], ii[(n+i)*is], i, limbs, w);
   
      MP_PTR_SWAP(ii[i*is],     *t1);
      MP_PTR_SWAP(ii[(n+i)*is], *t2);
   }

   mpir_fft_radix2_twiddle(ii, is, n/2, 2*w, t1, t2, ws, r, c, 2*rs);
#if 0
   ii += n * is;
   n /= 2;
   w += w;
   r += rs;
   rs += rs;
   goto start;
#else
   mpir_fft_radix2_twiddle(ii+n*is, is, n/2, 2*w, t1, t2, ws, r + rs, c, 2*rs);
#endif
}

void mpir_fft_trunc1_twiddle(mp_ptr * ii, mp_size_t is,
      mp_size_t n, mp_bitcnt_t w, mp_ptr * t1, mp_ptr * t2,
      mp_size_t ws, mp_size_t r, mp_size_t c, mp_size_t rs, mp_size_t trunc)
{
   mp_size_t i;
   mp_size_t limbs = (w*n)/GMP_LIMB_BITS;
   
   if (trunc == 2*n)
      mpir_fft_radix2_twiddle(ii, is, n, w, t1, t2, ws, r, c, rs);
   else if (trunc <= n)
   {
      for (i = 0; i < n; i++)
         mpn_add_n(ii[i*is], ii[i*is], ii[(i+n)*is], limbs + 1);
      
      mpir_fft_trunc1_twiddle(ii, is, n/2, 2*w, t1, t2, ws, r, c, 2*rs, trunc);
   } else
   {
      for (i = 0; i < n; i++) 
      {   
         mpir_fft_butterfly(*t1, *t2, ii[i*is], ii[(n+i)*is], i, limbs, w);
   
         MP_PTR_SWAP(ii[i*is],     *t1);
         MP_PTR_SWAP(ii[(n+i)*is], *t2);
      }

      mpir_fft_radix2_twiddle(ii, is, n/2, 2*w, t1, t2, ws, r, c, 2*rs);  
      mpir_fft_trunc1_twiddle(ii + n*is, is, n/2, 2*w, 
                                     t1, t2, ws, r + rs, c, 2*rs, trunc - n);
   }
}

void mpir_fft_mfa_trunc_sqrt2(mp_ptr * ii, mp_size_t n, 
                   mp_bitcnt_t w, mp_ptr * t1, mp_ptr * t2, 
                             mp_ptr * temp, mp_size_t n1, mp_size_t trunc)
{
   mp_size_t i, j, s;
   mp_size_t n2 = (2*n)/n1;
   mp_size_t trunc2 = (trunc - 2*n)/n1;
   mp_size_t limbs = (n*w)/GMP_LIMB_BITS;
   mp_bitcnt_t depth = 0;
   mp_bitcnt_t depth2 = 0;
   
   while ((((mp_size_t)1)<<depth) < n2) depth++;
   while ((((mp_size_t)1)<<depth2) < n1) depth2++;

   /* first half matrix fourier FFT : n2 rows, n1 cols */
   
   /* FFTs on columns */
   for (i = 0; i < n1; i++)
   {   
      /* relevant part of first layer of full sqrt2 FFT */
      if (w & 1)
      {
         for (j = i; j < trunc - 2*n; j+=n1) 
         {   
            if (j & 1)
               mpir_fft_butterfly_sqrt2(*t1, *t2, ii[j], ii[2*n+j], j, limbs, w, *temp);
            else
               mpir_fft_butterfly(*t1, *t2, ii[j], ii[2*n+j], j/2, limbs, w);     

            MP_PTR_SWAP(ii[j],     *t1);
            MP_PTR_SWAP(ii[2*n+j], *t2);
         }

         for ( ; j < 2*n; j+=n1)
         {
             if (i & 1)
                mpir_fft_adjust_sqrt2(ii[j + 2*n], ii[j], j, limbs, w, *temp); 
             else
                mpir_fft_adjust(ii[j + 2*n], ii[j], j/2, limbs, w); 
         }
      } else
      {
         for (j = i; j < trunc - 2*n; j+=n1) 
         {   
            mpir_fft_butterfly(*t1, *t2, ii[j], ii[2*n+j], j, limbs, w/2);
   
            MP_PTR_SWAP(ii[j],     *t1);
            MP_PTR_SWAP(ii[2*n+j], *t2);
         }

         for ( ; j < 2*n; j+=n1)
            mpir_fft_adjust(ii[j + 2*n], ii[j], j, limbs, w/2);
      }
   
      /* 
         FFT of length n2 on column i, applying z^{r*i} for rows going up in steps 
         of 1 starting at row 0, where z => w bits
      */
      
      mpir_fft_radix2_twiddle(ii + i, n1, n2/2, w*n1, t1, t2, w, 0, i, 1);
      for (j = 0; j < n2; j++)
      {
         mp_size_t s = mpir_revbin(j, depth);
         if (j < s) MP_PTR_SWAP(ii[i+j*n1], ii[i+s*n1]);
      }
   }
   
   /* FFTs on rows */
   for (i = 0; i < n2; i++)
   {
      mpir_fft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
      for (j = 0; j < n1; j++)
      {
         mp_size_t t = mpir_revbin(j, depth2);
         if (j < t) MP_PTR_SWAP(ii[i*n1+j], ii[i*n1+t]);
      }
   }
   
   /* second half matrix fourier FFT : n2 rows, n1 cols */
   ii += 2*n;

   /* FFTs on columns */
   for (i = 0; i < n1; i++)
   {   
      /*
         FFT of length n2 on column i, applying z^{r*i} for rows going up in steps 
         of 1 starting at row 0, where z => w bits
      */
      
      mpir_fft_trunc1_twiddle(ii + i, n1, n2/2, w*n1, t1, t2, w, 0, i, 1, trunc2);
      for (j = 0; j < n2; j++)
      {
         mp_size_t s = mpir_revbin(j, depth);
         if (j < s) MP_PTR_SWAP(ii[i+j*n1], ii[i+s*n1]);
      }
   }

   /* FFTs on relevant rows */
   for (s = 0; s < trunc2; s++)
   {
      i = mpir_revbin(s, depth);
      mpir_fft_radix2(ii + i*n1, n1/2, w*n2, t1, t2);
      
      for (j = 0; j < n1; j++)
      {
         mp_size_t t = mpir_revbin(j, depth2);
         if (j < t) MP_PTR_SWAP(ii[i*n1+j], ii[i*n1+t]);
      }
   }
}

void mpir_fft_mfa_trunc_sqrt2_outer(mp_ptr * ii, mp_size_t n, 
                   mp_bitcnt_t w, mp_ptr * t1, mp_ptr * t2, 
                             mp_ptr * temp, mp_size_t n1, mp_size_t trunc)
{
   mp_size_t i, j;
   mp_size_t n2 = (2*n)/n1;
   mp_size_t trunc2 = (trunc - 2*n)/n1;
   mp_size_t limbs = (n*w)/GMP_LIMB_BITS;
   mp_bitcnt_t depth = 0;
   mp_bitcnt_t depth2 = 0;
   
   while ((((mp_size_t)1)<<depth) < n2) depth++;
   while ((((mp_size_t)1)<<depth2) < n1) depth2++;

   /* first half matrix fourier FFT : n2 rows, n1 cols */
   
   /* FFTs on columns */
   for (i = 0; i < n1; i++)
   {   
      /* relevant part of first layer of full sqrt2 FFT */
      if (w & 1)
      {
         for (j = i; j < trunc - 2*n; j+=n1) 
         {   
            if (j & 1)
               mpir_fft_butterfly_sqrt2(*t1, *t2, ii[j], ii[2*n+j], j, limbs, w, *temp);
            else
               mpir_fft_butterfly(*t1, *t2, ii[j], ii[2*n+j], j/2, limbs, w);     

            MP_PTR_SWAP(ii[j],     *t1);
            MP_PTR_SWAP(ii[2*n+j], *t2);
         }

         for ( ; j < 2*n; j+=n1)
         {
             if (i & 1)
                mpir_fft_adjust_sqrt2(ii[j + 2*n], ii[j], j, limbs, w, *temp); 
             else
                mpir_fft_adjust(ii[j + 2*n], ii[j], j/2, limbs, w); 
         }
      } else
      {
         for (j = i; j < trunc - 2*n; j+=n1) 
         {   
            mpir_fft_butterfly(*t1, *t2, ii[j], ii[2*n+j], j, limbs, w/2);
   
            MP_PTR_SWAP(ii[j],     *t1);
            MP_PTR_SWAP(ii[2*n+j], *t2);
         }

         for ( ; j < 2*n; j+=n1)
            mpir_fft_adjust(ii[j + 2*n], ii[j], j, limbs, w/2);
      }
   
      /* 
         FFT of length n2 on column i, applying z^{r*i} for rows going up in steps 
         of 1 starting at row 0, where z => w bits
      */
      
      mpir_fft_radix2_twiddle(ii + i, n1, n2/2, w*n1, t1, t2, w, 0, i, 1);
      for (j = 0; j < n2; j++)
      {
         mp_size_t s = mpir_revbin(j, depth);
         if (j < s) MP_PTR_SWAP(ii[i+j*n1], ii[i+s*n1]);
      }
   }
      
   /* second half matrix fourier FFT : n2 rows, n1 cols */
   ii += 2*n;

   /* FFTs on columns */
   for (i = 0; i < n1; i++)
   {   
      /*
         FFT of length n2 on column i, applying z^{r*i} for rows going up in steps 
         of 1 starting at row 0, where z => w bits
      */
      
      mpir_fft_trunc1_twiddle(ii + i, n1, n2/2, w*n1, t1, t2, w, 0, i, 1, trunc2);
      for (j = 0; j < n2; j++)
      {
         mp_size_t s = mpir_revbin(j, depth);
         if (j < s) MP_PTR_SWAP(ii[i+j*n1], ii[i+s*n1]);
      }
   }
}
