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

#include <stdio.h>
#include <stdlib.h>
#include <mpir.h>
#include "gmp-impl.h"
#include "tests.h"

/* set p = 2^wn + 1 */
void set_p(mpz_t p, mp_size_t n, mp_bitcnt_t w)
{
   mpz_set_ui(p, 1);
   mpz_mul_2exp(p, p, n*w);
   mpz_add_ui(p, p, 1);
}

void ref_fft_butterfly_sqrt2(mpz_t s, mpz_t t, mpz_t i1, mpz_t i2, 
                                 mpz_t p, mp_size_t i, mp_size_t limbs, mp_size_t w)
{
   mpz_sub(t, i1, i2);
   mpz_mul_2exp(t, t, i*(w/2) + i/2);
   mpz_mul_2exp(s, t, 3*limbs*GMP_LIMB_BITS/4);
   mpz_mul_2exp(t, t, limbs*GMP_LIMB_BITS/4);
   mpz_sub(t, s, t);
   mpz_add(s, i1, i2);
   mpz_mod(s, s, p);
   mpz_mod(t, t, p);
}

void ref_ifft_butterfly_sqrt2(mpz_t s, mpz_t t, mpz_t i1, mpz_t i2, 
                                 mpz_t p, mp_size_t i, mp_size_t n, mp_size_t limbs, mp_size_t w)
{
   mpz_mul_2exp(s, i2, 2*n*w - i*(w/2) - 1 - i/2);
   mpz_mul_2exp(t, s, 3*limbs*GMP_LIMB_BITS/4);
   mpz_mul_2exp(s, s, limbs*GMP_LIMB_BITS/4);
   mpz_sub(i2, t, s);
   mpz_add(s, i1, i2);
   mpz_sub(t, i1, i2);
   mpz_mod(s, s, p);
   mpz_mod(t, t, p);
}

int
main(void)
{
    mp_size_t c, bits, j, k, n, w, limbs;
    mpz_t p, ma, mb, m2a, m2b, mn1, mn2;
    mp_limb_t * nn1, * nn2, * r1, * r2, * temp;
   
    gmp_randstate_t state;

    tests_start();
    fflush(stdout);

    gmp_randinit_default(state);

    mpz_init(p);
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(m2a);
    mpz_init(m2b);
    mpz_init(mn1);
    mpz_init(mn2);
   
    for (bits = GMP_LIMB_BITS; bits < 20*GMP_LIMB_BITS; bits += GMP_LIMB_BITS)
    {
        for (j = 1; j < 10; j++)
        {
            for (k = 1; k <= GMP_LIMB_BITS; k <<= 1)
            {
                n = bits/k;
                w = j*k;

                if ((w & 1) == 0) continue; /* w must be odd here */

                limbs = (n*w)/GMP_LIMB_BITS;

                for (c = 1; c < 2*n; c+=2)
                {
                    nn1 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    nn2 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    temp = malloc((limbs + 1)*sizeof(mp_limb_t));
                    r1 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    r2 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    mpir_random_fermat(nn1, state, limbs);
                    mpir_random_fermat(nn2, state, limbs);
                     
                    mpir_fermat_to_mpz(mn1, nn1, limbs);
                    mpir_fermat_to_mpz(mn2, nn2, limbs);
                    set_p(p, n, w);
            
                    mpir_fft_butterfly_sqrt2(r1, r2, nn1, nn2, c, limbs, w, temp);
                    mpir_fermat_to_mpz(m2a, r1, limbs);
                    mpir_fermat_to_mpz(m2b, r2, limbs);
                    
                    mpz_mod(m2a, m2a, p);
                    mpz_mod(m2b, m2b, p);
                    ref_fft_butterfly_sqrt2(ma, mb, mn1, mn2, p, c, limbs, w);

                    if (mpz_cmp(ma, m2a) != 0)
                    {
                        printf("FAIL:\n");
                        printf("fft_butterfly_sqrt2 error a\n");
                        printf("limbs = %ld\n", limbs);
                        printf("n = %ld, w = %ld, k = %ld, c = %ld\n", n, w, k, c);
                        gmp_printf("want %Zx\n\n", ma);
                        gmp_printf("got  %Zx\n", m2a);
                        abort();
                    }
                    if (mpz_cmp(mb, m2b) != 0)
                    {
                        printf("FAIL:\n");
                        printf("fft_butterfly_sqrt2 error b\n");
                        printf("limbs = %ld\n", limbs);
                        printf("n = %ld, w = %ld, k = %ld, c = %ld\n", n, w, k, c);
                        gmp_printf("want %Zx\n\n", mb);
                        gmp_printf("got  %Zx\n", m2b);
                        abort();
                    }
                    
                    free(temp);
                    free(nn1);
                    free(nn2);
                    free(r1);
                    free(r2);
                }
            }
        }
    }

    for (bits = GMP_LIMB_BITS; bits < 20*GMP_LIMB_BITS; bits += GMP_LIMB_BITS)
    {
        for (j = 1; j < 10; j++)
        {
            for (k = 1; k <= GMP_LIMB_BITS; k <<= 1)
            {
                n = bits/k;
                w = j*k;
                if ((w & 1) == 0) continue; /* w must be odd here */

                limbs = (n*w)/GMP_LIMB_BITS;

                for (c = 1; c < 2*n; c+=2)
                {
                    nn1 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    nn2 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    temp = malloc((limbs + 1)*sizeof(mp_limb_t));
                    r1 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    r2 = malloc((limbs + 1)*sizeof(mp_limb_t));
                    mpir_random_fermat(nn1, state, limbs);
                    mpir_random_fermat(nn2, state, limbs);
                     
                    mpir_fermat_to_mpz(mn1, nn1, limbs);
                    mpir_fermat_to_mpz(mn2, nn2, limbs);
                    set_p(p, n, w);
            
                    mpir_ifft_butterfly_sqrt2(r1, r2, nn1, nn2, c, limbs, w, temp);
                    mpir_fermat_to_mpz(m2a, r1, limbs);
                    mpir_fermat_to_mpz(m2b, r2, limbs);
                    
                    mpz_mod(m2a, m2a, p);
                    mpz_mod(m2b, m2b, p);
                    ref_ifft_butterfly_sqrt2(ma, mb, mn1, mn2, p, c, n, limbs, w);

                    if (mpz_cmp(ma, m2a) != 0)
                    {
                        printf("FAIL:\n");
                        printf("ifft_butterfly_sqrt2 error a\n");
                        printf("limbs = %ld\n", limbs);
                        printf("n = %ld, w = %ld, k = %ld, c = %ld\n", n, w, k, c);
                        gmp_printf("want %Zx\n\n", ma);
                        gmp_printf("got  %Zx\n", m2a);
                        abort();
                    }
                    if (mpz_cmp(mb, m2b) != 0)
                    {
                        printf("FAIL:\n");
                        printf("ifft_butterfly_sqrt2 error b\n");
                        printf("limbs = %ld\n", limbs);
                        printf("n = %ld, w = %ld, k = %ld, c = %ld\n", n, w, k, c);
                        gmp_printf("want %Zx\n\n", mb);
                        gmp_printf("got  %Zx\n", m2b);
                        abort();
                    }
                    free(temp);
                    free(nn1);
                    free(nn2);
                    free(r1);
                    free(r2);
                }
            }
        }
    }

    mpz_clear(p);
    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(m2a);
    mpz_clear(m2b);
    mpz_clear(mn1);
    mpz_clear(mn2);

    gmp_randclear(state);
    
    tests_end();
    return 0;
}
