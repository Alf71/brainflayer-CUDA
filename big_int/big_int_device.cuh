#pragma once

#include <stdint.h>
#include <cuda_runtime.h>



//----------------------------------------
// Helpers for big-endian <-> 32-bit words
//----------------------------------------

// 256-bit => c[8]
__device__  void loadBE256toWords(const uint8_t src[32], uint32_t c[8]);
__device__  void storeWordsToBE256(const uint32_t c[8], uint8_t dst[32]);

// 512-bit => c[16]
__device__  void loadBE512toWords(const uint8_t src[64], uint32_t c[16]);
__device__  void storeWordsToBE512(const uint32_t c[16], uint8_t dst[64]);

// 1024-bit => c[32]
__device__  void loadBE1024toWords(const uint8_t src[128], uint32_t c[32]);
__device__  void storeWordsToBE1024(const uint32_t c[32], uint8_t dst[128]);

// 2048-bit => c[64]
__device__  void loadBE2048toWords(const uint8_t src[256], uint32_t c[64]);
__device__  void storeWordsToBE2048(const uint32_t c[64], uint8_t dst[256]);

// 4096-bit => c[128]
__device__  void loadBE4096toWords(const uint8_t src[512], uint32_t c[128]);
__device__  void storeWordsToBE4096(const uint32_t c[128], uint8_t dst[512]);

// 8192-bit => c[256]
__device__  void loadBE8192toWords(const uint8_t src[1024], uint32_t c[256]);
__device__  void storeWordsToBE8192(const uint32_t c[256], uint8_t dst[1024]);

//----------------------------------------
//----------------------------------------
__device__  int add64to256_device(const uint32_t in[8],
    uint64_t val64,
    uint32_t out[8]);

__device__  int sub64from256_device(const uint32_t in[8],
    uint64_t val64,
    uint32_t out[8]);

//----------------------------------------
//----------------------------------------
__device__  int add64to512_device(const uint32_t in[16],
    uint64_t val64,
    uint32_t out[16]);

__device__  int sub64from512_device(const uint32_t in[16],
    uint64_t val64,
    uint32_t out[16]);

//----------------------------------------
//----------------------------------------
__device__  int add64to1024_device(const uint32_t in[32],
    uint64_t val64,
    uint32_t out[32]);

__device__  int sub64from1024_device(const uint32_t in[32],
    uint64_t val64,
    uint32_t out[32]);

//----------------------------------------
//----------------------------------------
__device__  int add64to2048_device(const uint32_t in[64],
    uint64_t val64,
    uint32_t out[64]);

__device__  int sub64from2048_device(const uint32_t in[64],
    uint64_t val64,
    uint32_t out[64]);

//----------------------------------------
//----------------------------------------
__device__  int add64to4096_device(const uint32_t in[128],
    uint64_t val64,
    uint32_t out[128]);

__device__  int sub64from4096_device(const uint32_t in[128],
    uint64_t val64,
    uint32_t out[128]);

//----------------------------------------
//----------------------------------------
__device__  int add64to8192_device(const uint32_t in[256],
    uint64_t val64,
    uint32_t out[256]);

__device__  int sub64from8192_device(const uint32_t in[256],
    uint64_t val64,
    uint32_t out[256]);

//----------------------------------------
//----------------------------------------
__device__  int sizeBE256_device(const uint32_t c[8]);

__device__  int sizeBE512_device(const uint32_t c[16]);

// 1024-bit
__device__  int sizeBE1024_device(const uint32_t c[32]);

// 2048-bit
__device__  int sizeBE2048_device(const uint32_t c[64]);

// 4096-bit
__device__  int sizeBE4096_device(const uint32_t c[128]);

// 8192-bit
__device__  int sizeBE8192_device(const uint32_t c[256]);

//----------------------------------------
//----------------------------------------
__device__  int cmp256_device(const uint32_t A[8],
    const uint32_t B[8]);

__device__  int cmp512_device(const uint32_t A[16],
    const uint32_t B[16]);
__device__  int cmp1024_device(const uint32_t A[32],
    const uint32_t B[32]);
__device__  int cmp2048_device(const uint32_t A[64],
    const uint32_t B[64]);
__device__  int cmp4096_device(const uint32_t A[128],
    const uint32_t B[128]);
__device__  int cmp8192_device(const uint32_t A[256],
    const uint32_t B[256]);


__device__ bool bump_key_256(uint8_t* __restrict__ prv32, uint64_t n, bool plus);

__device__ int bump_all_keys(uint8_t* __restrict__ prvKeys, int count, uint64_t n, bool plus, int* zeroz);

__device__ void mul_u64_u64_to_u128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi);


__device__ void add128to256_device(const uint32_t c[8], uint64_t lo, uint64_t hi, uint32_t r[8]);


__device__ void sub128from256_device(const uint32_t c[8], uint64_t lo, uint64_t hi, uint32_t r[8]);
// Done.
