#pragma once
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Initialise state
__device__  void SHA256Initialize(uint32_t s[8]);


__device__ void SHA256TransformPre(int level, uint32_t s[8], uint32_t* w, uint32_t* hashPre);

__device__ void SHA256TransformFromPre(const int level, uint32_t s[8], uint32_t* __restrict__ w, const uint32_t* __restrict__  hashPre);

// Perform SHA-256 transformations, process 64-byte chunks
__device__  void SHA256Transform(uint32_t s[8], uint32_t* __restrict__ w);

__device__ void SHA256(const uint8_t* __restrict__ msg, size_t len, uint8_t out[32]);

__device__
void SHA224(const uint8_t* __restrict__ msg, size_t len, uint8_t out[28]);

__device__ void hmac_sha256(const uint8_t* __restrict__ key, size_t key_len,
    const uint8_t* __restrict__ msg, size_t msg_len,
    uint8_t* out_mac);

__device__ void RIPEMD160Initialize(uint32_t s[5]);

__device__  void RIPEMD160Transform(uint32_t s[5], uint32_t* __restrict__  w);

__device__   void _GetHash160(const unsigned char* __restrict__  pubkey, int& keyLen, uint8_t* __restrict__ hash);


__device__   void _GetHash160Comp(unsigned char* pubkey, int& keyLen, uint8_t* hash);

__device__ void _GetHash160Comp_fast(uint64_t* x, uint8_t isOdd, uint8_t* hash);

// Both odd and !odd compressed hashes from same x. Writes to hash_odd (prefix 0x03) and hash_even (prefix 0x02).
__device__ void _GetHash160CompSym(uint64_t* x, uint8_t* hash_odd, uint8_t* hash_even);

__device__ void _GetHash160Uncomp_fast(uint64_t* x, uint64_t* y, uint8_t* hash);
__device__ void u64x4_to_pubkey65(uint64_t* px, uint64_t* py, unsigned char* out65);
__device__ void u64x4_to_x32(uint64_t* px, unsigned char* out32);
__device__ void keccak_eth64_tail20_bytes(const uint8_t* message, uint32_t* out5);
__device__ void keccak_eth64_tail20_from_xy(const uint64_t* px, const uint64_t* py, uint32_t* out5);


__device__   void _GetHash160ED(unsigned char* pubkey, int& keyLen, uint8_t* hash);




__device__ void _GetHash160(const uint32_t* x32, const uint32_t* y32, uint8_t* hash);

__device__  void _GetHash160P2SHCompFromHash(uint32_t* h, uint32_t* hash);

__device__  void _GetRMD160(const uint32_t* h, uint32_t* hash);

__device__ void MD5(const uint8_t* data, size_t len, uint8_t out16[16]);

__device__ uint32_t crc32_ieee(const uint8_t* data, size_t len);

__device__ void chacha20poly1305_encrypt(const uint8_t key[32], const uint8_t nonce[12], const uint8_t* plaintext, size_t pt_len, uint8_t* ciphertext, uint8_t tag[16]);


