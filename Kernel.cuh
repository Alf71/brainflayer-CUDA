#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lib/hash/GPUHash.cuh"
#include "lib/hash/sha3_ver3.cuh"
#include "lib/secp256k1/secp256k1_common.cuh"

using std::string;
using std::vector;

#ifdef _DEBUG
#define THREAD_STEPS 3
#define THREAD_STEPS_BRUTE 5
#define THREAD_STEPS_BRAIN 2
#else
#define THREAD_STEPS 32
#define THREAD_STEPS_BRUTE 64
#define THREAD_STEPS_BRAIN 16
#endif

#define ALPHABET_LEN 68
#define KERNEL_LAUNCH_BOUNDS

#ifdef __CUDA_ARCH__
#else
#define atomicAdd(x, n) ((uint64_t)x + (uint64_t)n)
#ifndef __noinline__
#define __noinline__
#endif
#endif

static constexpr uint8_t ENDO_TAG_BASE = 0xA0u;
static constexpr uint8_t ENDO_GROUP_STRIDE = 8u;
static constexpr uint8_t ENDO_GROUP_COMPRESSED = 0u;
static constexpr uint8_t ENDO_GROUP_SEGWIT = 1u;
static constexpr uint8_t ENDO_GROUP_UNCOMPRESSED = 2u;
static constexpr uint8_t ENDO_GROUP_ETH = 3u;
static constexpr uint8_t ENDO_GROUP_TAPROOT = 4u;
static constexpr uint8_t ENDO_GROUP_XPOINT = 5u;
static constexpr uint8_t ENDO_VARIANT_BASE_POS = 0u;
static constexpr uint8_t ENDO_VARIANT_BASE_NEG = 1u;
static constexpr uint8_t ENDO_VARIANT_ENDO1_POS = 2u;
static constexpr uint8_t ENDO_VARIANT_ENDO1_NEG = 3u;
static constexpr uint8_t ENDO_VARIANT_ENDO2_POS = 4u;
static constexpr uint8_t ENDO_VARIANT_ENDO2_NEG = 5u;
static constexpr uint32_t BRAIN_MASK_MAX_LEN = 64u;
static constexpr uint32_t BRAIN_MASK_MAX_CHARS = 4096u;

struct BrainMaskSpec {
    uint32_t len;
    uint32_t total_charsets;
    uint64_t total_candidates;
    uint16_t charset_offset[BRAIN_MASK_MAX_LEN];
    uint16_t charset_len[BRAIN_MASK_MAX_LEN];
    uint8_t chars[BRAIN_MASK_MAX_CHARS];
};

extern __constant__ uint32_t _NUM_TARGET_HASHES[1];
extern __constant__ uint32_t HASH_TARGET_WORDS[5];
extern __constant__ uint32_t HASH_TARGET_MASKS[5];
extern __constant__ uint32_t HASH_TARGET_LEN[1];
extern __constant__ uint32_t HASH_TARGET_ENABLED[1];
extern __constant__ __align__(8) uint8_t* _BLOOM_FILTER[100];

extern __device__ __align__(8) uint32_t* fingerprints_d[25];
extern __device__ __align__(8) size_t size_d[25];
extern __device__ __align__(8) size_t arrayLength_d[25];
extern __device__ __align__(8) size_t segmentCount_d[25];
extern __device__ __align__(8) size_t segmentCountLength_d[25];
extern __device__ __align__(8) size_t segmentLength_d[25];
extern __device__ __align__(8) size_t segmentLengthMask_d[25];

extern __constant__ __align__(8) uint32_t* fingerprints_d_Un[25];
extern __device__ __align__(8) size_t size_d_Un[25];
extern __device__ __align__(8) size_t arrayLength_d_Un[25];
extern __device__ __align__(8) size_t segmentCount_d_Un[25];
extern __constant__ __align__(8) size_t segmentCountLength_d_Un[25];
extern __constant__ __align__(8) size_t segmentLength_d_Un[25];
extern __constant__ __align__(8) size_t segmentLengthMask_d_Un[25];

extern __device__ __align__(8) uint16_t* fingerprints_d_Uc[25];
extern __device__ __align__(8) size_t size_d_Uc[25];
extern __device__ __align__(8) size_t arrayLength_d_Uc[25];
extern __device__ __align__(8) size_t segmentCount_d_Uc[25];
extern __device__ __align__(8) size_t segmentCountLength_d_Uc[25];
extern __device__ __align__(8) size_t segmentLength_d_Uc[25];
extern __device__ __align__(8) size_t segmentLengthMask_d_Uc[25];

extern __device__ __align__(8) uint8_t* fingerprints_d_Hc[25];
extern __device__ __align__(8) size_t size_d_Hc[25];
extern __device__ __align__(8) size_t arrayLength_d_Hc[25];
extern __device__ __align__(8) size_t segmentCount_d_Hc[25];
extern __device__ __align__(8) size_t segmentCountLength_d_Hc[25];
extern __device__ __align__(8) size_t segmentLength_d_Hc[25];
extern __device__ __align__(8) size_t segmentLengthMask_d_Hc[25];

extern __constant__ uint32_t _USE_BLOOM_FILTER[1];
extern __constant__ int _bloom_count[1];
extern __device__ int _xor_count[1];
extern __constant__ int _xor_un_count[1];
extern __device__ int _xor_uc_count[1];
extern __device__ int _xor_hc_count[1];
extern __device__ bool useBloom_d;
extern __device__ bool useXor_d;
extern __device__ bool useXorUn_d;
extern __device__ bool useXorUc_d;
extern __device__ bool useXorHc_d;

extern __constant__ __align__(64) uint8_t SECP_G65[65];

extern __device__ unsigned long long int d_resultsCount[1];
extern __device__ char (*d_foundStrings)[512];
extern __device__ unsigned char (*d_foundPrvKeys)[64];
extern __device__ uint32_t (*d_foundHash160)[20];
extern __device__ uint32_t (*d_len)[1];
extern __device__ uint32_t (*d_iter)[1];
extern __device__ uint8_t* d_type;
extern __device__ uint32_t* d_foundDerivations;
extern __device__ int64_t* d_round;

extern __device__ bool secp256_d;
extern __device__ bool ed25519_d;
extern __device__ bool compressed_dev;
extern __device__ bool uncompressed_dev;
extern __device__ bool segwit_dev;
extern __device__ bool taproot_dev;
extern __device__ bool ethereum_dev;
extern __device__ bool xpoint_dev;
extern __device__ bool solana_dev;
extern __device__ bool ton_dev;
extern __device__ bool ton_all_dev;
extern __device__ bool dot_dev;
extern __device__ bool aptos_dev;
extern __device__ bool sui_dev;
extern __device__ bool xrp_dev;
extern __device__ bool iota_dev;
extern __device__ bool icp_dev;
extern __device__ bool fil_dev;
extern __device__ bool xtz_dev;
extern __device__ bool endomorphism_dev;

extern __device__ uint64_t Seed;
extern __device__ uint64_t SeqStep;
extern __device__ uint32_t MAX_FOUNDS_DEV;
extern __device__ bool IS_HEX_DEV;
extern __device__ bool FULL_d;

extern std::atomic<uint64_t> false_positive;
extern bool STOP_THREAD;
extern uint32_t MAX_FOUNDS;
extern std::vector<std::thread> g_save_threads;
extern std::mutex g_save_threads_mutex;

typedef struct __align__(16) {
    uint8_t key[32];
    uint8_t chain_code[32];
} extended_private_key_t;

typedef struct __align__(16) {
    uint8_t key[64];
    uint8_t chain_code[32];
} extended_public_key_t;

typedef struct __align__(16) {
    uint64_t inner_H[8];
    uint64_t outer_H[8];
} hmac_sha512_precomp_t;

cudaError_t loadHashTarget(const uint32_t words[5], const uint32_t masks[5], uint32_t lenBytes, bool enabled);
cudaError_t cudaMemcpyToSymbol_BLOOM_FILTER(uint8_t* bloomFilterPtr, int count);
cudaError_t cudaMemcpyToSymbol_XOR(uint32_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h);
cudaError_t cudaMemcpyToSymbol_XORUn(uint32_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h);
cudaError_t cudaMemcpyToSymbol_XORUc(uint16_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h);
cudaError_t cudaMemcpyToSymbol_XORHc(uint8_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h);
cudaError_t loadWindow(unsigned int windowSize, unsigned int windows);

__global__ void setHEX();
__global__ void setFoundSize(uint32_t max_founds);
__global__ void SetSeqStep(uint64_t step_host);
__global__ void setFilterType(bool bloomUse, bool xorFilter, bool xorFilterUn, bool xorFilterUc, bool xorFilterHc);
__global__ void SetCurve(bool secp256, bool ed25519, bool compressed, bool uncompressed, bool segwit, bool taproot, bool ethereum, bool xpoint, bool solana, bool ton, bool ton_all, bool dot, bool aptos, bool sui, bool xrp, bool iota, bool icp, bool fil, bool xtz, bool endomorphism);

__global__ void workerBrain(bool* isResult, bool* buffResult, char* __restrict__ lines, const uint32_t* __restrict__ indexes, uint32_t indexes_size, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, uint64_t round, uint8_t brain_mode, const uint32_t* __restrict__ iterations, uint32_t iterations_size);
__global__ void workerBrain_seq(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, uint64_t round, uint8_t brain_mode, int mode, uint8_t* start_point_dev, int min_len, uint32_t iter);
__global__ KERNEL_LAUNCH_BOUNDS void workerPRIV(bool* isResult, bool* buffResult, char* __restrict__ lines, const uint32_t* __restrict__ indexes, uint32_t indexes_size, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, uint64_t round);
__global__ void advance_P0_kernel(uint64_t advance);
__global__ void compute_P0_H_kernel(const uint8_t* __restrict__ start_point, uint64_t step, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode);
cudaError_t cuda_vanity_set_step_increment(uint64_t step, const void* precPtr, size_t precPitch);
cudaError_t cuda_vanity_set_persistent_shift(uint64_t advance);
cudaError_t cuda_vanity_apply_persistent_shift(uint64_t* startx_buf, uint64_t* starty_buf, dim3 grid, dim3 block);
__global__ void precompute_vanity_starts_kernel(uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf, int thread_steps_pub);
__global__ void workerPRIV_seq_128(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_x(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_c(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_u(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_s(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_r(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_e(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void workerPRIV_seq_vanity_cusr(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub, uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf);
__global__ void buildMaskBatch(const BrainMaskSpec* __restrict__ spec, uint64_t candidate_start, char* __restrict__ lines, uint32_t* __restrict__ indexes, uint32_t count);

__global__ void shaPre(size_t prefixLen, uint32_t* hashPre);
__global__ void ecmult_big_create(secp256k1_gej* gej_temp, secp256k1_fe* z_ratio, secp256k1_ge_storage* precPtr, size_t precPitch, unsigned int bits);

__host__ void setSilentMode();
__host__ void wait_for_current_gpu_async_save_queue();
__host__ void flush_all_async_save_queues();
__host__ void shutdown_async_save_queues();
__host__ void push_save_cpu_postcheck_suppression();
__host__ void pop_save_cpu_postcheck_suppression();
__host__ void SaveResultBrain(FILE* file, uint32_t& Founds, bool save, vector<string> Der_list);
__host__ void SaveResultPRIV(FILE* file, uint32_t& Founds, bool save, vector<string> Der_list);

__device__ __host__ unsigned char* hexing(unsigned char* buf, size_t buf_sz, unsigned char* hexed, size_t hexed_sz);
__device__ unsigned char* unhex(unsigned char* str, size_t str_sz, unsigned char* unhexed, size_t unhexed_sz);
__device__ int lltoa(uint64_t* __restrict__ val, char* __restrict__ buf, const char* __restrict__ dict);
__device__ uint32_t SWAP256(uint32_t val);
__device__ uint64_t SWAP512(uint64_t val);
__device__ void md_pad_128(uint64_t* msg, long msgLen_bytes);
__device__ void sha256_process2(const uint32_t* W, uint32_t* digest);
__device__ void sha512_d(uint64_t* input, uint32_t length, uint64_t* hash);
__device__ void md_pad_128_swap(uint64_t* msg, long msgLen_bytes);
__device__ void sha512_swap(uint64_t* input, uint32_t length, uint64_t* hash);
__device__ void hmac_sha512_const(const uint32_t* key, const uint32_t* message, uint32_t* output);
__device__ void hmac_sha512_const_precompute(const uint32_t* key, hmac_sha512_precomp_t* ctx);
__device__ void hmac_sha512_const_precomp(const hmac_sha512_precomp_t* ctx, const uint32_t* message, uint32_t* output);
__device__ void sha256_d(const uint32_t* pass, int pass_len, uint32_t* hash);
__device__ void sha256_swap_64(const uint32_t* pass, uint32_t* hash);
__device__ void random32(uint32_t* output, int size);
__device__ size_t TweakTaproot_batch(uint8_t* __restrict__ out, const uint8_t* __restrict__ pub_uncomp, int count, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch);
__device__ void TweakTaproot(uint8_t* __restrict__ out, const uint8_t* __restrict__ pub_uncomp, const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch);

__device__ bool pubkey_to_hash_ton(const uint8_t* public_key, const char* type, uint8_t* out, size_t out_len = 32);
__device__ int icp_principal_from_ed25519(const uint8_t pub_ed25519[32], uint8_t out_principal[29]);
__device__ int icp_principal_from_secp256k1(const uint8_t pub_secp256k1[65], uint8_t out_principal[29]);
__device__ int icp_account_identifier(const uint8_t principal29[29], const uint8_t* subaccount32, uint8_t out_account32[32]);
__device__ bool checkHashEth(const unsigned char d_hash[32]);
__device__ bool bloom_chk_hash160(const unsigned char* bloom, const uint32_t* h);
__device__ uint64_t fnv1a_64(const uint8_t* buffer, size_t length);
__device__ bool checkHash(const uint32_t hash[5]);

__device__ void hardened_private_child_from_private(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t hardened_child_number);
__device__ void normal_private_child_from_private(const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, const extended_private_key_t* parent, extended_private_key_t* child, uint32_t normal_child_number);
__device__ void normal_private_child_from_private_cached_pub(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t normal_child_number, const uint8_t* cached_serialized_pub);
__device__ void normal_private_child_from_private_save_pub(const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch, const extended_private_key_t* parent, extended_private_key_t* child, uint32_t normal_child_number, uint8_t* out_serialized_pub);
__device__ void normal_private_child_from_private_cached_pub_precomp(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t normal_child_number, const uint8_t* cached_serialized_pub, const hmac_sha512_precomp_t* hctx);
__device__ void hardened_private_child_from_private_precomp(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t hardened_child_number, const hmac_sha512_precomp_t* hctx);
__device__ void hardened_private_child_from_private_ed25519(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t hardened_child_number);
__device__ void hardened_private_child_from_private_ed25519_precomp(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t hardened_child_number, const hmac_sha512_precomp_t* hctx);
__device__ void ed25519_bip32_ckd_priv_hardened(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t i_hardened);
__device__ void ed25519_bip32_ckd_priv_normal(const extended_private_key_t* parent, extended_private_key_t* child, uint32_t i_normal);

#endif
