extern __device__  const ge25519 __align__(16) ge25519_basepoint;

/*
	d
*/

extern __device__  const bignum25519 __align__(16) ge25519_ecd;

extern __device__  const bignum25519 __align__(16) ge25519_ec2d;

/*
	sqrt(-1)
*/

extern __device__  const bignum25519 __align__(16) ge25519_sqrtneg1;

extern __device__  const ge25519_niels __align__(16) ge25519_niels_sliding_multiples[32];
