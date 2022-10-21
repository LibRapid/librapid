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

int
main(void)
{
    mp_bitcnt_t depth, w;
    int iters;

    gmp_randstate_t state;

    tests_start();
    fflush(stdout);

    gmp_randinit_default(state);

    for (iters = 0; iters < 100; iters++)
    {
        for (depth = 6; depth <= 18; depth++)
        {
            for (w = 1; w <= 2; w++)
            {
                mp_size_t n = (((mp_limb_t)1)<<depth);
                mp_bitcnt_t bits = n*w;
                mp_size_t int_limbs = bits/GMP_LIMB_BITS;
                mp_size_t j;
                mp_limb_t c, * i1, * i2, * r1, * r2, * tt;
        
                i1 = malloc(6*(int_limbs+1)*sizeof(mp_limb_t));
                i2 = i1 + int_limbs + 1;
                r1 = i2 + int_limbs + 1;
                r2 = r1 + int_limbs + 1;
                tt = r2 + int_limbs + 1;

                mpir_random_fermat(i1, state, int_limbs);
                mpir_random_fermat(i2, state, int_limbs);
                mpn_normmod_2expp1(i1, int_limbs);
                mpn_normmod_2expp1(i2, int_limbs);

                mpn_mulmod_Bexpp1(r2, i1, i2, n * w / GMP_LIMB_BITS, tt);
                c = 2*i1[int_limbs] + i2[int_limbs];
                c = mpn_mulmod_2expp1_basecase(r1, i1, i2, c, int_limbs*GMP_LIMB_BITS, tt);
            
                for (j = 0; j < int_limbs; j++)
                {
                    if (r1[j] != r2[j]) 
                    {
                        gmp_printf("error in limb %ld, %Mx != %Mx\n", j, r1[j], r2[j]);
                        abort();
                    }
                }

                if (c != r2[int_limbs])
                {
                    gmp_printf("error in limb %ld, %Mx != %Mx\n", j, c, r2[j]);
                    abort();
                }

                free(i1);
            }
        }
    }

    /* test squaring */
    for (iters = 0; iters < 100; iters++)
    {
        for (depth = 6; depth <= 18; depth++)
        {
            for (w = 1; w <= 2; w++)
            {
                mp_size_t n = (((mp_limb_t)1)<<depth);
                mp_bitcnt_t bits = n*w;
                mp_size_t int_limbs = bits/GMP_LIMB_BITS;
                mp_size_t j;
                mp_limb_t c, * i1, * r1, * r2, * tt;
        
                i1 = malloc(5*(int_limbs+1)*sizeof(mp_limb_t));
                r1 = i1 + int_limbs + 1;
                r2 = r1 + int_limbs + 1;
               tt = r2 + int_limbs + 1;

                mpir_random_fermat(i1, state, int_limbs);
                mpn_normmod_2expp1(i1, int_limbs);
                
                mpn_mulmod_Bexpp1(r2, i1, i1, n * w / GMP_LIMB_BITS, tt);
                c = i1[int_limbs] + 2*i1[int_limbs];
                c = mpn_mulmod_2expp1_basecase(r1, i1, i1, c, int_limbs*GMP_LIMB_BITS, tt);
            
                for (j = 0; j < int_limbs; j++)
                {
                    if (r1[j] != r2[j]) 
                    {
                        gmp_printf("error in limb %ld, %Mx != %Mx\n", j, r1[j], r2[j]);
                        abort();
                    }
                }

                if (c != r2[int_limbs])
                {
                    gmp_printf("error in limb %ld, %Mx != %Mx\n", j, c, r2[j]);
                    abort();
                }

                free(i1);
            }
        }
    }

    gmp_randclear(state);
    
    tests_end();
    return 0;
}
