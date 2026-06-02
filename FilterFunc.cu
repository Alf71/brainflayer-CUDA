#include "Kernel.cuh"
#ifdef _WIN32
#include "uint128_t.cuh"
#else
#include "uint128_t_linux.cuh"
#endif

// Builds a 32-bit fingerprint from a 64-bit hash value.
__device__ __forceinline__ uint32_t fingerprint(const uint64_t hash) {
	return (uint32_t)(hash ^ (hash >> 32));
}

// Builds a 16-bit fingerprint used by ultra-compressed XOR filters.
__device__ __forceinline__ uint16_t fingerprintUc(const uint64_t hash) {
	return (uint16_t)(hash ^ (hash >> 32));
}

// Builds an 8-bit fingerprint used by hyper-compressed XOR filters.
__device__ __forceinline__ uint8_t fingerprintHc(const uint64_t hash) {
	return (uint8_t)(hash ^ (hash >> 32));
}

__device__ uint64_t rng_splitmix64(uint64_t* seed)
{
	uint64_t z = (*seed += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t getHashFromHash(uint64_t hash, int index, int xors)
{
	uint128_t x = (uint128_t)hash * (uint128_t)segmentCountLength_d[xors];
	uint64_t h = (uint64_t)(x >> 64);
	h += index * segmentLength_d[xors];
	uint64_t hh = hash & ((1ULL << 32) - 1);
	if (index < 3)
	{
		// index 0: right shift by 36; index 1: right shift by 18; index 2: no shift
		h ^= (size_t)((hh >> (36 - 18 * index)) & segmentLengthMask_d[xors]);
	}
	return h;
}

__device__ __forceinline__ uint64_t getHashFromHashUn(uint64_t hash, int index, int xors)
{
	uint128_t x = (uint128_t)hash * (uint128_t)segmentCountLength_d_Un[xors];
	uint64_t h = (uint64_t)(x >> 64);
	h += index * segmentLength_d_Un[xors];
	uint64_t hh = hash & ((1ULL << 32) - 1);
	// index 0: right shift by 36; index 1: right shift by 18; index 2: no shift
	if (index < 3)
	{
		h ^= (size_t)((hh >> (36 - 18 * index)) & segmentLengthMask_d_Un[xors]);
	}
	return h;
}

__device__ __forceinline__ uint64_t getHashFromHashUc(uint64_t hash, int index, int xors)
{
	uint128_t x = (uint128_t)hash * (uint128_t)segmentCountLength_d_Uc[xors];
	uint64_t h = (uint64_t)(x >> 64);
	h += index * segmentLength_d_Uc[xors];
	uint64_t hh = hash & ((1ULL << 32) - 1);
	// index 0: right shift by 36; index 1: right shift by 18; index 2: no shift
	if (index < 3)
	{
		h ^= (size_t)((hh >> (36 - 18 * index)) & segmentLengthMask_d_Uc[xors]);
	}
	return h;
}

__device__ __forceinline__ uint64_t getHashFromHashHc(uint64_t hash, int index, int xors)
{
	uint128_t x = (uint128_t)hash * (uint128_t)segmentCountLength_d_Hc[xors];
	uint64_t h = (uint64_t)(x >> 64);
	h += index * segmentLength_d_Hc[xors];
	uint64_t hh = hash & ((1ULL << 32) - 1);
	// index 0: right shift by 36; index 1: right shift by 18; index 2: no shift
	if (index < 3)
	{
		h ^= (size_t)((hh >> (36 - 18 * index)) & segmentLengthMask_d_Hc[xors]);
	}
	return h;
}

// MurmurHash3 finalizer used as a lightweight 64-bit mixer.
__device__ __forceinline__ uint64_t murmur64(uint64_t h) {
	//uint64_t h = (hash.m_hi ^ hash.m_lo);
	h ^= h >> 33;
	h *= UINT64_C(0xff51afd7ed558ccd);
	h ^= h >> 33;
	h *= UINT64_C(0xc4ceb9fe1a85ec53);
	h ^= h >> 33;
	return h;
}

// Checks membership in the compressed XOR filter set.
__device__ __forceinline__ bool Contain(const uint64_t& item, int xors) {
	const uint64_t hash = murmur64(item + Seed);
	uint32_t f = fingerprint(hash);

	const size_t base = (size_t)__umul64hi(hash, (uint64_t)segmentCountLength_d[xors]);
	const size_t seg_len = segmentLength_d[xors];
	const size_t seg_mask = segmentLengthMask_d[xors];
	const uint64_t hh = hash & ((1ULL << 32) - 1);

	size_t h0 = base;
	size_t h1 = (base + seg_len) ^ (size_t)((hh >> 18) & seg_mask);
	size_t h2 = (base + (seg_len << 1)) ^ (size_t)(hh & seg_mask);
	size_t h3 = base + seg_len * 3;

	f ^= __ldg(&fingerprints_d[xors][h0]);
	f ^= __ldg(&fingerprints_d[xors][h1]);
	f ^= __ldg(&fingerprints_d[xors][h2]);
	f ^= __ldg(&fingerprints_d[xors][h3]);

	return (f == 0);
}

// Checks membership in the ultra-compressed XOR filter set.
__device__ __forceinline__ bool ContainUc(const uint64_t& item, int xors) {
	const uint64_t hash = murmur64(item + Seed);
	uint16_t f = fingerprintUc(hash);

	const size_t base = (size_t)__umul64hi(hash, (uint64_t)segmentCountLength_d_Uc[xors]);
	const size_t seg_len = segmentLength_d_Uc[xors];
	const size_t seg_mask = segmentLengthMask_d_Uc[xors];
	const uint64_t hh = hash & ((1ULL << 32) - 1);

	size_t h0 = base;
	size_t h1 = (base + seg_len) ^ (size_t)((hh >> 18) & seg_mask);
	size_t h2 = (base + (seg_len << 1)) ^ (size_t)(hh & seg_mask);
	size_t h3 = base + seg_len * 3;

	f ^= __ldg(&fingerprints_d_Uc[xors][h0]);
	f ^= __ldg(&fingerprints_d_Uc[xors][h1]);
	f ^= __ldg(&fingerprints_d_Uc[xors][h2]);
	f ^= __ldg(&fingerprints_d_Uc[xors][h3]);

	return (f == 0);
}

// Checks membership in the hyper-compressed XOR filter set.
__device__ __forceinline__ bool ContainHc(const uint64_t& item, int xors) {
	const uint64_t hash = murmur64(item + Seed);
	uint8_t f = fingerprintHc(hash);

	const size_t base = (size_t)__umul64hi(hash, (uint64_t)segmentCountLength_d_Hc[xors]);
	const size_t seg_len = segmentLength_d_Hc[xors];
	const size_t seg_mask = segmentLengthMask_d_Hc[xors];
	const uint64_t hh = hash & ((1ULL << 32) - 1);

	size_t h0 = base;
	size_t h1 = (base + seg_len) ^ (size_t)((hh >> 18) & seg_mask);
	size_t h2 = (base + (seg_len << 1)) ^ (size_t)(hh & seg_mask);
	size_t h3 = base + seg_len * 3;

	f ^= __ldg(&fingerprints_d_Hc[xors][h0]);
	f ^= __ldg(&fingerprints_d_Hc[xors][h1]);
	f ^= __ldg(&fingerprints_d_Hc[xors][h2]);
	f ^= __ldg(&fingerprints_d_Hc[xors][h3]);

	return (f == 0);
}

// Performs one half-hash probe against an uncompressed XOR filter.
__device__ __forceinline__ bool ContainUnHalfHashRaw(
	const uint64_t hash,
	const uint32_t* __restrict__ fingerprints,
	const size_t segment_count_length,
	const size_t segment_length,
	const size_t segment_mask)
{
	uint32_t f = fingerprint(hash);

	const size_t base = (size_t)__umul64hi(hash, (uint64_t)segment_count_length);
	const uint64_t hh = hash & ((1ULL << 32) - 1);

	const size_t h0 = base;
	const size_t h1 = (base + segment_length) ^ (size_t)((hh >> 18) & segment_mask);
	const size_t h2 = (base + (segment_length << 1)) ^ (size_t)(hh & segment_mask);
	const size_t h3 = base + segment_length * 3;

	f ^= __ldg(&fingerprints[h0]);
	f ^= __ldg(&fingerprints[h1]);
	f ^= __ldg(&fingerprints[h2]);
	f ^= __ldg(&fingerprints[h3]);
	return f == 0;
}

// Convenience wrapper that reads uncompressed filter metadata from constant memory.
__device__ __forceinline__ bool ContainUnHalfHash(const uint64_t hash, int xors) {
	return ContainUnHalfHashRaw(
		hash,
		fingerprints_d_Un[xors],
		segmentCountLength_d_Un[xors],
		segmentLength_d_Un[xors],
		segmentLengthMask_d_Un[xors]);
}

// Checks both 64-bit halves of a packed 128-bit value in uncompressed XOR filters.
__device__ __forceinline__ bool ContainUn(const uint128_t& item, int xors) {
	const uint64_t hash_hi = murmur64(item.m_hi + Seed);
	if (!ContainUnHalfHash(hash_hi, xors)) {
		return false;
	}
	const uint64_t hash_lo = murmur64(item.m_lo + Seed);
	return ContainUnHalfHash(hash_lo, xors);
}



#define BLOOM_SIZE (512*1024*1024)
#define BLOOM_SET_BIT(N) (bloom[(N)>>3] = bloom[(N)>>3] | (1<<((N)&7)))
#define BLOOM_GET_BIT(N) ( ( bloom[(N)>>3]>>((N)&7) )&1)


#define BH00(N) (N[0])
#define BH01(N) (N[1])
#define BH02(N) (N[2])
#define BH03(N) (N[3])
#define BH04(N) (N[4])

#define BH05(N) (N[0]<<16|N[1]>>16)
#define BH06(N) (N[1]<<16|N[2]>>16)
#define BH07(N) (N[2]<<16|N[3]>>16)
#define BH08(N) (N[3]<<16|N[4]>>16)
#define BH09(N) (N[4]<<16|N[0]>>16)

#define BH10(N) (N[0]<< 8|N[1]>>24)
#define BH11(N) (N[1]<< 8|N[2]>>24)
#define BH12(N) (N[2]<< 8|N[3]>>24)
#define BH13(N) (N[3]<< 8|N[4]>>24)
#define BH14(N) (N[4]<< 8|N[0]>>24)

#define BH15(N) (N[0]<<24|N[1]>> 8)
#define BH16(N) (N[1]<<24|N[2]>> 8)
#define BH17(N) (N[2]<<24|N[3]>> 8)
#define BH18(N) (N[3]<<24|N[4]>> 8)
#define BH19(N) (N[4]<<24|N[0]>> 8)

// Reads one Bloom filter bit through read-only cache.
__device__ __forceinline__ uint32_t bloom_get_bit_ro(const unsigned char* bloom, const uint32_t n) {
	const uint8_t byte = __ldg(reinterpret_cast<const uint8_t*>(bloom) + (n >> 3));
	return (uint32_t)((byte >> (n & 7u)) & 1u);
}

// Converts ETH-style 32-byte Keccak output into HASH160-like 5-word tail and checks filters.
__device__ bool checkHashEth(const unsigned char d_hash[32]) {
	uint32_t hash[5];
	for (int h = 12, i = 0; i < 5; ++i) {
		hash[i] = (d_hash[h++]) | ((d_hash[h++] << 8) & 0x0000ff00) | ((d_hash[h++] << 16) & 0x00ff0000) | ((d_hash[h++] << 24) & 0xff000000);
	}
	return checkHash(hash);
}

// Checks all 20 derived Bloom indexes for one 160-bit hash value.
__device__ __forceinline__ bool bloom_chk_hash160(const unsigned char* bloom, const uint32_t* h) {
	const uint32_t h0 = h[0];
	const uint32_t h1 = h[1];
	const uint32_t h2 = h[2];
	const uint32_t h3 = h[3];
	const uint32_t h4 = h[4];

#define BLOOM_CHECK(_idx) do { if (bloom_get_bit_ro(bloom, (_idx)) == 0u) return false; } while (0)
	BLOOM_CHECK(h0);
	BLOOM_CHECK(h1);
	BLOOM_CHECK(h2);
	BLOOM_CHECK(h3);
	BLOOM_CHECK(h4);
	BLOOM_CHECK((h0 << 16) | (h1 >> 16));
	BLOOM_CHECK((h1 << 16) | (h2 >> 16));
	BLOOM_CHECK((h2 << 16) | (h3 >> 16));
	BLOOM_CHECK((h3 << 16) | (h4 >> 16));
	BLOOM_CHECK((h4 << 16) | (h0 >> 16));
	BLOOM_CHECK((h0 << 8) | (h1 >> 24));
	BLOOM_CHECK((h1 << 8) | (h2 >> 24));
	BLOOM_CHECK((h2 << 8) | (h3 >> 24));
	BLOOM_CHECK((h3 << 8) | (h4 >> 24));
	BLOOM_CHECK((h4 << 8) | (h0 >> 24));
	BLOOM_CHECK((h0 << 24) | (h1 >> 8));
	BLOOM_CHECK((h1 << 24) | (h2 >> 8));
	BLOOM_CHECK((h2 << 24) | (h3 >> 8));
	BLOOM_CHECK((h3 << 24) | (h4 >> 8));
	BLOOM_CHECK((h4 << 24) | (h0 >> 8));
#undef BLOOM_CHECK
	return true;
}

// FNV-1a helper used to reduce 20-byte hash material into one 64-bit probe value.
__device__ __forceinline__ uint64_t fnv1a_64(const uint8_t* buffer, size_t length) {
	uint64_t h = 0xcbf29ce484222325ULL;
	for (size_t i = 0; i < length; ++i) {
		h ^= buffer[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

// Fast path for uncompressed XOR filters using packed 128-bit hash representation.
__device__ __forceinline__ bool checkHashXorUnPacked(const uint32_t hash[5]) {
	// build packed 128-bit key directly from 5x32-bit hash words (aligned-friendly, no byte memcpy)
	const uint32_t mix = hash[4];
	const uint32_t b0 = mix & 0xFFu;
	const uint32_t b1 = (mix >> 8) & 0xFFu;
	const uint32_t b2 = (mix >> 16) & 0xFFu;
	const uint32_t b3 = (mix >> 24) & 0xFFu;

	const uint32_t p0 = (hash[0] & 0x00FFFFFFu) | ((((hash[0] >> 24) & 0xFFu) & b0) << 24);
	const uint32_t p1 = (hash[1] & 0x00FFFFFFu) | ((((hash[1] >> 24) & 0xFFu) & b1) << 24);
	const uint32_t p2 = (hash[2] & 0x00FFFFFFu) | ((((hash[2] >> 24) & 0xFFu) & b2) << 24);
	const uint32_t p3 = (hash[3] & 0x00FFFFFFu) | ((((hash[3] >> 24) & 0xFFu) & b3) << 24);

	const int xor_count = _xor_un_count[0];
	if (xor_count <= 0) {
		return false;
	}

	const uint64_t packed_lo = ((uint64_t)p1 << 32) | (uint64_t)p0;
	const uint64_t packed_hi = ((uint64_t)p3 << 32) | (uint64_t)p2;
	const uint64_t seed = Seed;
	const uint64_t hash_hi = murmur64(packed_hi + seed);
	const uint64_t hash_lo = murmur64(packed_lo + seed);

	if (xor_count == 1) {
		const uint32_t* fp = fingerprints_d_Un[0];
		const size_t seg_count_len = segmentCountLength_d_Un[0];
		const size_t seg_len = segmentLength_d_Un[0];
		const size_t seg_mask = segmentLengthMask_d_Un[0];
		if (!ContainUnHalfHashRaw(hash_hi, fp, seg_count_len, seg_len, seg_mask)) {
			return false;
		}
		return ContainUnHalfHashRaw(hash_lo, fp, seg_count_len, seg_len, seg_mask);
	}

	for (int i = 0; i < xor_count; ++i) {
		const uint32_t* fp = fingerprints_d_Un[i];
		const size_t seg_count_len = segmentCountLength_d_Un[i];
		const size_t seg_len = segmentLength_d_Un[i];
		const size_t seg_mask = segmentLengthMask_d_Un[i];
		if (!ContainUnHalfHashRaw(hash_hi, fp, seg_count_len, seg_len, seg_mask)) {
			continue;
		}
		if (ContainUnHalfHashRaw(hash_lo, fp, seg_count_len, seg_len, seg_mask)) {
			return true;
		}
	}
	return false;
}

// Fast path for Bloom-only checks.
__device__ __forceinline__ bool checkHashBloomOnly(const uint32_t hash[5]) {
	const int bloom_count = _bloom_count[0];
	if (bloom_count == 1) {
		return bloom_chk_hash160(_BLOOM_FILTER[0], hash);
	}
	for (int i = 0; i < bloom_count; ++i) {
		if (bloom_chk_hash160(_BLOOM_FILTER[i], hash)) {
			return true;
		}
	}
	return false;
}

__device__ __forceinline__ bool checkHashTargetPrefix(const uint32_t hash[5]) {
	uint32_t diff = 0u;
	diff |= (hash[0] ^ HASH_TARGET_WORDS[0]) & HASH_TARGET_MASKS[0];
	diff |= (hash[1] ^ HASH_TARGET_WORDS[1]) & HASH_TARGET_MASKS[1];
	diff |= (hash[2] ^ HASH_TARGET_WORDS[2]) & HASH_TARGET_MASKS[2];
	diff |= (hash[3] ^ HASH_TARGET_WORDS[3]) & HASH_TARGET_MASKS[3];
	diff |= (hash[4] ^ HASH_TARGET_WORDS[4]) & HASH_TARGET_MASKS[4];
	return diff == 0u;
}

__device__ bool checkHash(const uint32_t hash[5]) {
	if (FULL_d)
	{
		return true;
	}
	const bool useHashTarget = (HASH_TARGET_ENABLED[0] != 0u);
	if (useHashTarget && !checkHashTargetPrefix(hash)) {
		return false;
	}
	const bool useBloom = useBloom_d;
	const bool useXor = useXor_d;
	const bool useXorUn = useXorUn_d;
	const bool useXorUc = useXorUc_d;
	const bool useXorHc = useXorHc_d;
	const bool anyFilter = useBloom || useXor || useXorUn || useXorUc || useXorHc;
	if (useHashTarget && !anyFilter) {
		return true;
	}

	// hot path for "-xu only" runs
	if (useXorUn && !useBloom && !useXor && !useXorUc && !useXorHc) {
		return checkHashXorUnPacked(hash);
	}
	// hot path for "-bf only" runs
	if (useBloom && !useXorUn && !useXor && !useXorUc && !useXorHc) {
		return checkHashBloomOnly(hash);
	}

	if (useBloom)
	{
		if (checkHashBloomOnly(hash)) {
			return true;
		}
	}
	if (useXor || useXorUc || useXorHc)
	{
		const uint8_t* hash_bytes = reinterpret_cast<const uint8_t*>(hash);
		const uint64_t hash64 = fnv1a_64(hash_bytes, 20);
		const int xor_count = useXor ? _xor_count[0] : 0;
		const int xor_uc_count = useXorUc ? _xor_uc_count[0] : 0;
		const int xor_hc_count = useXorHc ? _xor_hc_count[0] : 0;

		if (xor_count > 0)
		{
			for (int i = 0; i < xor_count; ++i)
			{
				if (Contain(hash64, i))
				{
					return true;
				}
			}
		}

		if (xor_uc_count > 0)
		{
			for (int i = 0; i < xor_uc_count; ++i)
			{
				if (ContainUc(hash64, i))
				{
					return true;
				}
			}
		}

		if (xor_hc_count > 0)
		{
			for (int i = 0; i < xor_hc_count; ++i)
			{
				if (ContainHc(hash64, i))
				{
					return true;
				}
			}
		}
	}
	if (useXorUn)
	{
		if (checkHashXorUnPacked(hash)) {
			return true;
		}
	}
	return false;
}
