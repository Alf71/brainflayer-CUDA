#ifndef ED25519_H
#define ED25519_H
#include <cstdint>
#include <stdlib.h>
//#include "ed25519-hash-custom.h"


#if defined(__cplusplus)
extern "C" {
#endif
//#include "ed25519-donna.h"

typedef unsigned char ed25519_signature[64];
typedef unsigned char ed25519_public_key[32];
typedef unsigned char ed25519_secret_key[32];

typedef unsigned char curved25519_key[32];


__device__ void set_scalar();
__device__ void set_hash();
__device__ void set_le();
__device__ void ed25519_publickey(const ed25519_secret_key sk, ed25519_public_key pk);
__device__ void ed25519_publickey_batch(const uint8_t* __restrict__ sk, uint8_t* __restrict__ pk, int n);
__device__ void ed25519_key_to_pub(const ed25519_secret_key sk, ed25519_public_key pk);
__device__ void ed25519_key_to_pub_batch(const uint8_t* __restrict__ sk, uint8_t* __restrict__ pk, int n);
__device__ int ed25519_sign_open(const unsigned char *m, size_t mlen, const ed25519_public_key pk, const ed25519_signature RS);
__device__ void ed25519_sign(const unsigned char *m, size_t mlen, const ed25519_secret_key sk, const ed25519_public_key pk, ed25519_signature RS);

__device__ int ed25519_sign_open_batch(const unsigned char **m, size_t *mlen, const unsigned char **pk, const unsigned char **RS, size_t num, int *valid);

__device__ void ed25519_randombytes_unsafe(void *out, size_t count);

__device__ void curved25519_scalarmult_basepoint(curved25519_key pk, const curved25519_key e);

__device__ void add_modL_from_bytes(uint8_t out32[32], const uint8_t inX[32], const uint8_t inY[32]);

#if defined(__cplusplus)
}
#endif


#endif // ED25519_H
