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

int
main(void)
{
    mp_bitcnt_t bits;
    mp_size_t j, k, n, w, limbs;
    mp_limb_t * nn;
    mpz_t p, m1, m2;

    gmp_randstate_t state;

    tests_start();
    fflush(stdout);

    gmp_randinit_default(state);

    mpz_init(m1);
    mpz_init(m2);
    mpz_init(p);

    /* normalisation mod p = 2^wn + 1 where B divides nw and n is a power of 2 */
    for (bits = GMP_LIMB_BITS; bits < 32*GMP_LIMB_BITS; bits += GMP_LIMB_BITS)
    {
        for (j = 1; j < 32; j++)
        {
            for (k = 1; k <= GMP_NUMB_BITS; k <<= 1)
            {
                n = bits/k;
                w = j*k;
                limbs = (n*w)/GMP_LIMB_BITS;
            
                nn = malloc((limbs + 1)*sizeof(mp_limb_t));
                mpn_rrandom(nn, state, limbs + 1);
                mpir_fermat_to_mpz(m1, nn, limbs);
                set_p(p, n, w);
            
                mpn_normmod_2expp1(nn, limbs);
                mpir_fermat_to_mpz(m2, nn, limbs);
                mpz_mod(m1, m1, p);

                if (mpz_cmp(m1, m2) != 0)
                {
                    printf("FAIL:\n");
                    printf("mpn_normmod_2expp1 error\n");
                    gmp_printf("want %Zx\n\n", m1);
                    gmp_printf("got  %Zx\n", m2);
                    abort();
                }

                free(nn);
            }
        }
    }

    mpz_clear(m2);
    mpz_clear(m1);
    mpz_clear(p);

    gmp_randclear(state);
    
    tests_end();
    return 0;
}
