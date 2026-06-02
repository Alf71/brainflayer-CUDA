#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>



/*** Keccak-f[1600] ***/
__device__  void keccakf(void* state);



/** The sponge-based hash construction. **/
__device__  void hashing(
	uint8_t* out,
	size_t outlen,
	uint8_t const* in,
	size_t inlen,
	size_t rate,
	uint8_t delim
);


__device__  void keccak(const char* __restrict__ message, int message_len, unsigned char* __restrict__ output, int output_len);






// ---------------- Keccak-f[1600] -----------------------------------------



__device__  void keccak_f1600(uint64_t state[25]);

// ---------------- SHA3-256 (Keccak) --------------------------------------

__device__  void sha3_256(const char* __restrict__ message, int message_len, unsigned char* __restrict__ output);

// ---------------- BLAKE2b-256 --------------------------------------------



__device__  void Blake2b_256(const uint8_t* data, size_t len, uint8_t out[32]);

__device__ void Blake2b_224(const uint8_t* data, size_t len, uint8_t out[28]);

__device__ void Blake2b_160(const uint8_t* data, size_t len, uint8_t out[20]);
