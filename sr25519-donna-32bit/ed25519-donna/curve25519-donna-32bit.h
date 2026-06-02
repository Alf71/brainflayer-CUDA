#pragma once
#include <cstdint>
typedef uint32_t bignum25519[10];
typedef uint32_t bignum25519align16[12];


/* out = in */
__device__   void
curve25519_copy(bignum25519 out, const bignum25519 in);

/* out = a + b */
__device__   void
curve25519_add(bignum25519 out, const bignum25519 a, const bignum25519 b);

__device__   void
curve25519_add_after_basic(bignum25519 out, const bignum25519 a, const bignum25519 b);

__device__   void
curve25519_add_reduce(bignum25519 out, const bignum25519 a, const bignum25519 b);



/* out = a - b */
__device__   void
curve25519_sub(bignum25519 out, const bignum25519 a, const bignum25519 b);

/* out = a - b, where a is the result of a basic op (add,sub) */
__device__   void
curve25519_sub_after_basic(bignum25519 out, const bignum25519 a, const bignum25519 b);

__device__   void
curve25519_sub_reduce(bignum25519 out, const bignum25519 a, const bignum25519 b);

/* out = -a */
__device__   void
curve25519_neg(bignum25519 out, const bignum25519 a);

/* out = a * b */
#define curve25519_mul_noinline curve25519_mul

__device__  void
curve25519_mul(bignum25519 out, const bignum25519 a, const bignum25519 b);

/* out = in*in */
__device__  void
curve25519_square(bignum25519 out, const bignum25519 in);


/* out = in ^ (2 * count) */
__device__  void
curve25519_square_times(bignum25519 out, const bignum25519 in, int count);

/* Take a little-endian, 32-byte number and expand it into polynomial form */
__device__  void
curve25519_expand(bignum25519 out, const unsigned char in[32]);

/* Take a fully reduced polynomial form number and contract it into a
 * little-endian, 32-byte array
 */
__device__  void
curve25519_contract(unsigned char out[32], const bignum25519 in);


/* out = (flag) ? in : out */
__device__   void
curve25519_move_conditional_bytes(uint8_t out[96], const uint8_t in[96], uint32_t flag);

/* if (iswap) swap(a, b) */
__device__   void
curve25519_swap_conditional(bignum25519 a, bignum25519 b, uint32_t iswap);
