// MT19937 RNG from https://github.com/bstatcomp/RandomCL/blob/master/generators/mt19937.cl

/**
@file

Implements Mersenne twister generator.

  M. Matsumoto, T. Nishimura, Mersenne twister: a 623-dimensionally equidistributed uniform
pseudo-random number generator, ACM Transactions on Modeling and Computer Simulation (TOMACS) 8 (1)
(1998) 3–30.
                                                                                                                    */

#define RNG32

#define MT19937_FLOAT_MULTI   2.3283064365386962890625e-10f
#define MT19937_DOUBLE2_MULTI 2.3283064365386962890625e-10
#define MT19937_DOUBLE_MULTI  5.4210108624275221700372640e-20

#define MT19937_N          624
#define MT19937_M          397
#define MT19937_MATRIX_A   0x9908b0df /* constant vector a */
#define MT19937_UPPER_MASK 0x80000000 /* most significant w-r bits */
#define MT19937_LOWER_MASK 0x7fffffff /* least significant r bits */

/**
State of MT19937 RNG.
*/
typedef struct {
    uint mt[MT19937_N]; /* the array for the state vector  */
    int mti;
} mt19937_state;

/**
Generates a random 32-bit unsigned integer using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_uint(state) _mt19937_uint(&state)
uint _mt19937_uint(mt19937_state *state) {
    uint y;
    uint mag01[2] = {0x0, MT19937_MATRIX_A};
    /* mag01[x] = x * MT19937_MATRIX_A  for x=0,1 */

    if (state->mti < MT19937_N - MT19937_M) {
        y = (state->mt[state->mti] & MT19937_UPPER_MASK) |
            (state->mt[state->mti + 1] & MT19937_LOWER_MASK);
        state->mt[state->mti] = state->mt[state->mti + MT19937_M] ^ (y >> 1) ^ mag01[y & 0x1];
    } else if (state->mti < MT19937_N - 1) {
        y = (state->mt[state->mti] & MT19937_UPPER_MASK) |
            (state->mt[state->mti + 1] & MT19937_LOWER_MASK);
        state->mt[state->mti] =
          state->mt[state->mti + (MT19937_M - MT19937_N)] ^ (y >> 1) ^ mag01[y & 0x1];
    } else {
        y = (state->mt[MT19937_N - 1] & MT19937_UPPER_MASK) | (state->mt[0] & MT19937_LOWER_MASK);
        state->mt[MT19937_N - 1] = state->mt[MT19937_M - 1] ^ (y >> 1) ^ mag01[y & 0x1];
        state->mti               = 0;
    }
    y = state->mt[state->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}
/**
Generates a random 32-bit unsigned integer using MT19937 RNG.

This is alternative implementation of MT19937 RNG, that generates 32 values in single call.

@param state State of the RNG to use.
*/
#define mt19937_loop_uint(state) _mt19937_loop_uint(&state)
uint _mt19937_loop_uint(mt19937_state *state) {
    uint y;
    uint mag01[2] = {0x0, MT19937_MATRIX_A};
    /* mag01[x] = x * MT19937_MATRIX_A  for x=0,1 */

    if (state->mti >= MT19937_N) {
        int kk;

        for (kk = 0; kk < MT19937_N - MT19937_M; kk++) {
            y = (state->mt[kk] & MT19937_UPPER_MASK) | (state->mt[kk + 1] & MT19937_LOWER_MASK);
            state->mt[kk] = state->mt[kk + MT19937_M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < MT19937_N - 1; kk++) {
            y = (state->mt[kk] & MT19937_UPPER_MASK) | (state->mt[kk + 1] & MT19937_LOWER_MASK);
            state->mt[kk] = state->mt[kk + (MT19937_M - MT19937_N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (state->mt[MT19937_N - 1] & MT19937_UPPER_MASK) | (state->mt[0] & MT19937_LOWER_MASK);
        state->mt[MT19937_N - 1] = state->mt[MT19937_M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        state->mti = 0;
    }

    y = state->mt[state->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}

/**
Seeds MT19937 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator
(thread).
*/
void mt19937_seed(mt19937_state *state, uint s) {
    state->mt[0] = s;
    uint mti;
    for (mti = 1; mti < MT19937_N; mti++) {
        state->mt[mti] = 1812433253 * (state->mt[mti - 1] ^ (state->mt[mti - 1] >> 30)) + mti;

        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt19937[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
    }
    state->mti = mti;
}

/**
Generates a random 64-bit unsigned integer using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_ulong(state) ((((ulong)mt19937_uint(state)) << 32) | mt19937_uint(state))

/**
Generates a random float using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_float(state) (mt19937_uint(state) * MT19937_FLOAT_MULTI)

/**
Generates a random double using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_double(state) (mt19937_ulong(state) * MT19937_DOUBLE_MULTI)

/**
Generates a random double using MT19937 RNG. Generated using only 32 random bits.

@param state State of the RNG to use.
*/
#define mt19937_double2(state) (mt19937_uint(state) * MT19937_DOUBLE2_MULTI)

#define RANDOM_IMPL(TYPE)                                                                          \
    __kernel void fillRandom_##TYPE(__global TYPE *data,                                           \
                                    int64_t elements,                                              \
                                    TYPE lower,                                                    \
                                    TYPE upper,                                                    \
                                    __global int64_t *seeds,                                       \
                                    int64_t numSeeds) {                                            \
        int64_t gid       = get_global_id(0);                                                      \
        int64_t seedIndex = gid % numSeeds;                                                        \
        mt19937_state state;                                                                       \
        mt19937_seed(&state, seeds[seedIndex]);                                                    \
                                                                                                   \
        for (int64_t i = gid; i < elements; i += get_global_size(0)) {                             \
            data[i] = (TYPE)(mt19937_double(state) * (upper - lower) + lower);                     \
        }                                                                                          \
                                                                                                   \
        /* Change the seed for the next iteration */                                               \
        seeds[seedIndex] = mt19937_ulong(state);                                                   \
    }

RANDOM_IMPL(int8_t)
RANDOM_IMPL(uint8_t)
RANDOM_IMPL(int16_t)
RANDOM_IMPL(uint16_t)
RANDOM_IMPL(int32_t)
RANDOM_IMPL(uint32_t)
RANDOM_IMPL(int64_t)
RANDOM_IMPL(uint64_t)
RANDOM_IMPL(float)
RANDOM_IMPL(double)
