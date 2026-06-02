#ifndef __SR25519_HASH_H__
#define __SR25519_HASH_H__

#include "sha2.h"

typedef SHA512_CTX sr25519_hash_context;

__device__  void
sr25519_hash_init(sr25519_hash_context* ctx);

__device__  void
sr25519_hash_update(sr25519_hash_context* ctx, const uint8_t* in, size_t inlen);


__device__  void
sr25519_hash_final(sr25519_hash_context* ctx, uint8_t* hash);


__device__  void
sr25519_hash(uint8_t* hash, const uint8_t* in, size_t inlen);



#endif
