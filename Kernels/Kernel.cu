

#include "Kernel.cuh"








#include "lib/secp256k1/secp256k1_common.cuh"
#include "lib/secp256k1/secp256k1_field.cuh"
#include "lib/secp256k1/secp256k1_prec8.cuh"
#include "lib/secp256k1/secp256k1_group.cuh"
#include "lib/secp256k1/secp256k1_scalar.cuh"
#include "lib/secp256k1/secp256k1.cuh"
#include "lib/secp256k1/secp256k1_batch_impl.cuh"
#include "sr25519-donna-32bit/ed25519-donna/ed25519.h"

extern __device__ uint64_t rng_splitmix64(uint64_t* seed);

__device__ void random32(uint32_t* output, int size)
{
	uint64_t x = clock64()
		^ ((uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 0x9e3779b97f4a7c15ULL)
		^ ((uint64_t)blockIdx.y << 32);
	int words = size / 4;
	for (int i = 0; i < words; ++i) {
		x ^= x >> 12;
		x ^= x << 25;
		x ^= x >> 27;
		output[i] = (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
	}
}

__constant__ uint8_t TAG_TAPTWEAK[32] = {
	0xe8,0x0f,0xe1,0x63,0x9c,0x9c,0xa0,0x50,0xe3,0xaf,0x1b,0x39,0xc1,0x43,0xc6,0x3e,
	0x42,0x9c,0xbc,0xeb,0x15,0xd9,0x40,0xfb,0xb5,0xc5,0xa1,0xf4,0xaf,0x57,0xc5,0xe9
};

__device__ __forceinline__ uint32_t tap_be32(const uint8_t* p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

__device__ __inline__ void sha256_taptweak_px(const uint8_t px[32], uint8_t out[32])
{
	uint32_t h[8];
	SHA256Initialize(h);
	uint32_t w0[16];
#pragma unroll
	for (int i = 0; i < 8; ++i) {
		w0[i] = tap_be32(&TAG_TAPTWEAK[4 * i]);
		w0[i + 8] = tap_be32(&TAG_TAPTWEAK[4 * i]);
	}
	SHA256Transform(h, w0);

	uint32_t w1[16];
#pragma unroll
	for (int i = 0; i < 8; ++i) {
		w1[i] = tap_be32(&px[4 * i]);
	}
	w1[8] = 0x80000000U;
#pragma unroll
	for (int i = 9; i < 15; ++i) {
		w1[i] = 0;
	}
	w1[15] = 0x00000300U;
	SHA256Transform(h, w1);

#pragma unroll
	for (int i = 0; i < 8; ++i) {
		out[4 * i + 0] = (uint8_t)(h[i] >> 24);
		out[4 * i + 1] = (uint8_t)(h[i] >> 16);
		out[4 * i + 2] = (uint8_t)(h[i] >> 8);
		out[4 * i + 3] = (uint8_t)h[i];
	}
}

__device__ size_t TweakTaproot_batch(
	uint8_t* __restrict__ out,
	const uint8_t* __restrict__ pub_uncomp,
	int count,
	const secp256k1_ge_storage* __restrict__ precPtr,
	size_t precPitch)
{
	secp256k1_scratch scr;
	int produced = 0;

#pragma unroll
	for (int i = 0; i < count; ++i) {
		const uint8_t* p = &pub_uncomp[i * 65];
		if (p[0] != 0x04) {
			scr.gej[i].infinity = 1;
			continue;
		}

		secp256k1_fe x, y;
		(void)secp256k1_fe_set_b32(&x, p + 1);
		(void)secp256k1_fe_set_b32(&y, p + 33);
		secp256k1_ge pnt;
		secp256k1_ge_set_xy(&pnt, &x, &y);

		secp256k1_fe_normalize_var(&pnt.y);
		if (secp256k1_fe_is_odd(&pnt.y)) {
			secp256k1_ge pneg = pnt;
			secp256k1_fe_negate(&pneg.y, &pneg.y, 1);
			pnt = pneg;
		}

		uint8_t px[32], h[32];
		secp256k1_fe_normalize_var(&pnt.x);
		secp256k1_fe_get_b32(px, &pnt.x);
		sha256_taptweak_px(px, h);

		secp256k1_scalar t;
		secp256k1_scalar_set_b32(&t, h, NULL);
		secp256k1_gej tg;
#ifdef ECMULT_BIG_TABLE
		int windowLimit = WINDOWS_SIZE_CONST[0];
		unsigned int wlimit = ECMULT_WINDOW_SIZE_CONST[0];
		secp256k1_ecmult_big(&tg, &t, precPtr, precPitch, windowLimit, wlimit);
#else
		secp256k1_ecmult_gen(&tg, &t);
#endif

		secp256k1_gej_add_ge_var(&scr.gej[i], &tg, &pnt, NULL);
		if (!scr.gej[i].infinity) {
			scr.fe_in[produced++] = scr.gej[i].z;
		}
	}

	if (produced > 0) {
		secp256k1_fe_inv_all_var(produced, scr.fe_out, scr.fe_in);
	}

	int used = 0;
#pragma unroll
	for (int i = 0; i < count; ++i) {
		uint8_t* xo = &out[i * 32];
		if (scr.gej[i].infinity) {
#pragma unroll
			for (int k = 0; k < 32; ++k) {
				xo[k] = 0;
			}
			continue;
		}

		secp256k1_ge a;
		secp256k1_ge_set_gej_zinv(&a, &scr.gej[i], &scr.fe_out[used++]);
		secp256k1_fe_normalize_var(&a.y);
		if (secp256k1_fe_is_odd(&a.y)) {
			secp256k1_ge_neg(&a, &a);
		}
		secp256k1_fe_normalize_var(&a.x);
		secp256k1_fe_get_b32(xo, &a.x);
	}

	return (size_t)produced;
}

__device__ void TweakTaproot(
	uint8_t* __restrict__ out,
	const uint8_t* __restrict__ pub_uncomp,
	const secp256k1_ge_storage* __restrict__ precPtr,
	size_t precPitch)
{
	__align__(16) secp256k1_scratch3 scr;
	const uint8_t* p = &pub_uncomp[0];
	if (p[0] != 0x04) {
		scr.gej.infinity = 1;
		return;
	}

	__align__(16) secp256k1_fe x, y;
	(void)secp256k1_fe_set_b32(&x, p + 1);
	(void)secp256k1_fe_set_b32(&y, p + 33);
	__align__(16) secp256k1_ge point;
	secp256k1_ge_set_xy(&point, &x, &y);

	secp256k1_fe_normalize_var(&point.y);
	if (secp256k1_fe_is_odd(&point.y)) {
		secp256k1_ge neg = point;
		secp256k1_fe_negate(&neg.y, &neg.y, 1);
		point = neg;
	}

	__align__(16) uint8_t px[32], h[32];
	secp256k1_fe_normalize_var(&point.x);
	secp256k1_fe_get_b32(px, &point.x);
	sha256_taptweak_px(px, h);

	__align__(16) secp256k1_scalar tweak;
	secp256k1_scalar_set_b32(&tweak, h, NULL);
	__align__(16) secp256k1_gej tg;
#ifdef ECMULT_BIG_TABLE
	int windowLimit = WINDOWS_SIZE_CONST[0];
	unsigned int wlimit = ECMULT_WINDOW_SIZE_CONST[0];
	secp256k1_ecmult_big(&tg, &tweak, precPtr, precPitch, windowLimit, wlimit);
#else
	secp256k1_ecmult_gen(&tg, &tweak);
#endif

	secp256k1_gej_add_ge_var(&scr.gej, &tg, &point, NULL);
	if (scr.gej.infinity) {
#pragma unroll
		for (int k = 0; k < 32; ++k) {
			out[k] = 0;
		}
		return;
	}

	scr.fe_in = scr.gej.z;
	secp256k1_fe_inv_all_var(1, &scr.fe_out, &scr.fe_in);
	__align__(16) secp256k1_ge result;
	secp256k1_ge_set_gej_zinv(&result, &scr.gej, &scr.fe_out);
	secp256k1_fe_normalize_var(&result.y);
	if (secp256k1_fe_is_odd(&result.y)) {
		secp256k1_ge_neg(&result, &result);
	}
	secp256k1_fe_normalize_var(&result.x);
	secp256k1_fe_get_b32(out, &result.x);
}






 __constant__ uint32_t _NUM_TARGET_HASHES[1] = { 0 };
 __constant__ uint32_t HASH_TARGET_WORDS[5] = { 0 };
 __constant__ uint32_t HASH_TARGET_MASKS[5] = { 0 };
 __constant__ uint32_t HASH_TARGET_LEN[1] = { 0 };
 __constant__ uint32_t HASH_TARGET_ENABLED[1] = { 0 };
__constant__ __align__(8) uint8_t* _BLOOM_FILTER[100] = { 0 };
//__device__ xorbinaryfusefilter_lowmem4wise::XorBinaryFuseFilter<uint128_t, uint32_t>* xors;
__device__ __align__(8) uint32_t* fingerprints_d[25] = { 0 };
__device__ __align__(8) size_t size_d[25] = { 0 };
__device__ __align__(8) size_t arrayLength_d[25] = { 0 };
__device__ __align__(8) size_t segmentCount_d[25] = { 0 };
__device__ __align__(8) size_t segmentCountLength_d[25] = { 0 };
__device__ __align__(8) size_t segmentLength_d[25] = { 0 };
__device__ __align__(8) size_t segmentLengthMask_d[25] = { 0 };

__constant__ __align__(8) uint32_t* fingerprints_d_Un[25] = { 0 };
__device__ __align__(8) size_t size_d_Un[25] = { 0 };
__device__ __align__(8) size_t arrayLength_d_Un[25] = { 0 };
__device__ __align__(8) size_t segmentCount_d_Un[25] = { 0 };
__constant__ __align__(8) size_t segmentCountLength_d_Un[25] = { 0 };
__constant__ __align__(8) size_t segmentLength_d_Un[25] = { 0 };
__constant__ __align__(8) size_t segmentLengthMask_d_Un[25] = { 0 };

__device__ __align__(8) uint16_t* fingerprints_d_Uc[25] = { 0 };
__device__ __align__(8) size_t size_d_Uc[25] = { 0 };
__device__ __align__(8) size_t arrayLength_d_Uc[25] = { 0 };
__device__ __align__(8) size_t segmentCount_d_Uc[25] = { 0 };
__device__ __align__(8) size_t segmentCountLength_d_Uc[25] = { 0 };
__device__ __align__(8) size_t segmentLength_d_Uc[25] = { 0 };
__device__ __align__(8) size_t segmentLengthMask_d_Uc[25] = { 0 };

__device__ __align__(8) uint8_t* fingerprints_d_Hc[25] = { 0 };
__device__ __align__(8) size_t size_d_Hc[25] = { 0 };
__device__ __align__(8) size_t arrayLength_d_Hc[25] = { 0 };
__device__ __align__(8) size_t segmentCount_d_Hc[25] = { 0 };
__device__ __align__(8) size_t segmentCountLength_d_Hc[25] = { 0 };
__device__ __align__(8) size_t segmentLength_d_Hc[25] = { 0 };
__device__ __align__(8) size_t segmentLengthMask_d_Hc[25] = { 0 };

 __constant__ uint32_t _USE_BLOOM_FILTER[1] = { 0 };
__constant__ int _bloom_count[1] = { 0 };
__device__ int _xor_count[1] = { 0 };
__constant__ int _xor_un_count[1] = { 0 };
__device__ int _xor_uc_count[1] = { 0 };
__device__ int _xor_hc_count[1] = { 0 };

__device__ bool useBloom_d = false;
__device__ bool useXor_d = false;
__device__ bool useXorUn_d = false;
__device__ bool useXorUc_d = false;
__device__ bool useXorHc_d = false;

__device__ unsigned long long int d_resultsCount[1] = { 0 };

__constant__ __align__(64) uint8_t SECP_G65[65] = {
  0x04,
  0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,
  0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98,
  0x48,0x3a,0xda,0x77,0x26,0xa3,0xc4,0x65,0x5d,0xa4,0xfb,0xfc,0x0e,0x11,0x08,0xa8,
  0xfd,0x17,0xb4,0x48,0xa6,0x85,0x54,0x19,0x9c,0x47,0xd0,0x8f,0xfb,0x10,0xd4,0xb8
};

__device__ char           (*d_foundStrings)[512];
__device__ unsigned char  (*d_foundPrvKeys)[64];
__device__ uint32_t(*d_foundHash160)[20];
__device__ uint32_t(*d_len)[1];
__device__ uint32_t(*d_iter)[1];
__device__ uint8_t* d_type;
__device__ uint32_t* d_foundDerivations;
__device__ int64_t* d_round;

__device__ bool secp256_d = false;
__device__ bool ed25519_d = false;

__device__ bool compressed_dev = false;
__device__ bool uncompressed_dev = false;
__device__ bool segwit_dev = false;
__device__ bool taproot_dev = false;
__device__ bool ethereum_dev = false;
__device__ bool xpoint_dev = false;
__device__ bool solana_dev = false;
__device__ bool ton_dev = false;
__device__ bool ton_all_dev = false;
__device__ bool dot_dev = false;
__device__ bool aptos_dev = false;
__device__ bool sui_dev = false;
__device__ bool xrp_dev = false;
__device__ bool iota_dev = false;
__device__ bool	icp_dev = false;
__device__ bool	fil_dev = false;
__device__ bool	xtz_dev = false;
__device__ bool endomorphism_dev = false;

__device__ uint64_t Seed = 0;
__device__ uint64_t SeqStep = 1;

__device__ uint32_t MAX_FOUNDS_DEV = 0;

__device__ bool IS_HEX_DEV = false;
__device__ bool FULL_d = false;

__global__ void setHEX()
{
	IS_HEX_DEV = true;
}

__global__ void SetSeqStep(uint64_t step_host)
{
	SeqStep = step_host;
}

__global__ void SetCurve(bool secp256, bool ed25519, bool compressed, bool uncompressed, bool segwit, bool taproot, bool ethereum, bool xpoint, bool solana, bool ton, bool ton_all, bool dot, bool aptos, bool sui, bool xrp, bool iota, bool icp, bool fil, bool xtz, bool endomorphism) {

	secp256_d = secp256;
	ed25519_d = ed25519;

	compressed_dev = compressed;
	uncompressed_dev = uncompressed;
	endomorphism_dev = endomorphism;
	segwit_dev = segwit;
	taproot_dev = taproot;
	ethereum_dev = ethereum;
	xpoint_dev = xpoint;
	solana_dev = solana;
	ton_dev = ton;
	ton_all_dev = ton_all;
	dot_dev = dot;
	aptos_dev = aptos;
	sui_dev = sui;
	xrp_dev = xrp;
	iota_dev = iota;
	icp_dev = icp;
	fil_dev = fil;
	xtz_dev = xtz;

}

__global__ void setFoundSize(uint32_t max_founds)
{
	MAX_FOUNDS_DEV = max_founds;
	
}


__device__ __host__ unsigned char* hexing(unsigned char* buf, size_t buf_sz, unsigned char* hexed, size_t hexed_sz) {
	static const char hex_chars[] = "0123456789abcdef";
	const size_t n = (buf_sz < (hexed_sz / 2)) ? buf_sz : (hexed_sz / 2);
	for (size_t i = 0; i < n; ++i) {
		hexed[2 * i] = hex_chars[(buf[i] >> 4) & 0xF];
		hexed[2 * i + 1] = hex_chars[buf[i] & 0xF];
	}
	if (hexed_sz > 0) {
		hexed[2 * n] = '\0';
	}
	return hexed;
}


__constant__ __align__(16) unsigned char unhex_tab[80] = {
  0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
  0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};


__device__ unsigned char*
unhex(unsigned char* str, size_t str_sz,
	unsigned char* unhexed, size_t unhexed_sz) {
	size_t j = 0;
	const size_t n = (str_sz < (unhexed_sz * 2)) ? str_sz : (unhexed_sz * 2);
	for (size_t i = 0; (i + 1) < n; i += 2, ++j) {
		unhexed[j] = (unhex_tab[str[i] & 0x4f] & 0xf0) |
			(unhex_tab[str[i + 1] & 0x4f] & 0x0f);
	}
	return unhexed;
}




// Kernel entry point: ecmult_big_create.
__global__ void ecmult_big_create(secp256k1_gej* gej_temp, secp256k1_fe* z_ratio, secp256k1_ge_storage* precPtr, size_t precPitch, unsigned int bits) {
	int64_t tIx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tIx != 0) {
		return;
	}
	unsigned int windows;
	size_t window_size;
	size_t i, row;
	secp256k1_fe  fe_zinv;
	secp256k1_ge  ge_temp;
	secp256k1_ge  ge_window_one = secp256k1_ge_const_g;
	secp256k1_gej gej_window_base;

	/* We +1 to account for a possible high 1 bit after converting the privkey to signed digit form.    */
	/* This means our table reaches to 257 bits even though the privkey scalar is at most 256 bits.     */
	//unsigned int bits = (unsigned int)ECMULT_WINDOW_SIZE;
	windows = (256 / bits) + 1;
	window_size = (1 << (bits - 1));
	WINDOWS = windows;
	WINDOW_SIZE = WINDOW_SIZE;
	ECMULT_WINDOW_SIZE = bits;
	//size_t total_size = (256 / bits) * window_size + (1 << (256 % bits));

	//windows = WINDOWS;
	//window_size = WINDOW_SIZE;
	//bits = ECMULT_WINDOW_SIZE;

	/* Total number of required point storage elements.                                 */
	/* This differs from the (windows * window_size) because the last row can be shrunk */
	/*   as it only needs to extend enough to include a possible 1 in the 257th bit.    */
	//total_size = (256 / bits) * window_size + (1 << (256 % bits));

	//rtn->gej_temp = (secp256k1_gej*)checked_malloc(&ctx->error_callback, sizeof(secp256k1_gej) * window_size);
	//rtn->z_ratio = (secp256k1_fe*)checked_malloc(&ctx->error_callback, sizeof(secp256k1_fe) * window_size);

	/************ Precomputed Table Initialization ************/
	secp256k1_gej_set_ge(&gej_window_base, &ge_window_one);

	/* This is the same for all windows.    */
	secp256k1_fe_set_int(&(z_ratio[0]), 0);


	for (row = 0; row < windows; row++) {
		/* The last row is a bit smaller, only extending to include the 257th bit. */
		window_size = (row == windows - 1 ? (1 << (256 % bits)) : (1 << (bits - 1)));

		/* The base element of each row is 2^bits times the previous row's base. */
		if (row > 0) {
			for (i = 0; i < bits; i++) {
				secp256k1_gej_double_var(&gej_window_base, &gej_window_base, NULL);
			}
		}
		gej_temp[0] = gej_window_base;

		/* The base element is also our "one" value for this row.   */
		/* If we are at offset 2^X, adding "one" should add 2^X.    */
		secp256k1_ge_set_gej(&ge_window_one, &gej_window_base);


		/* Repeated + 1s to fill the rest of the row.   */

		/* We capture the Z ratios between consecutive points for quick Z inversion.    */
		/*   gej_temp[i-1].z * z_ratio[i] => gej_temp[i].z                              */
		/* This means that z_ratio[i] = (gej_temp[i-1].z)^-1 * gej_temp[i].z            */
		/* If we know gej_temp[i].z^-1, we can get gej_temp[i-1].z^1 using z_ratio[i]   */
		/* Visually:                                    */
		/* i            0           1           2       */
		/* gej_temp     a           b           c       */
		/* z_ratio     NaN      (a^-1)*b    (b^-1)*c    */
		for (i = 1; i < window_size; i++) {
			secp256k1_gej_add_ge_var(&(gej_temp[i]), &(gej_temp[i - 1]), &ge_window_one, &(z_ratio[i]));
		}


		/* An unpacked version of secp256k1_ge_set_table_gej_var() that works   */
		/*   element by element instead of requiring a secp256k1_ge *buffer.    */

		/* Invert the last Z coordinate manually.   */
		i = window_size - 1;
		secp256k1_fe_inv(&fe_zinv, &(gej_temp[i].z));
		secp256k1_ge_set_gej_zinv(&ge_temp, &(gej_temp[i]), &fe_zinv);
		//secp256k1_ge_to_storage(&(prec[row][i]), &ge_temp);
		secp256k1_ge_storage* ROW_PREC = (secp256k1_ge_storage*)((char*)precPtr + row * precPitch) + i;
		secp256k1_ge_to_storage(ROW_PREC, &ge_temp);

		/* Use the last element's known Z inverse to determine the previous' Z inverse. */
		for (; i > 0; i--) {
			/* fe_zinv = (gej_temp[i].z)^-1                 */
			/* (gej_temp[i-1].z)^-1 = z_ratio[i] * fe_zinv  */
			secp256k1_fe_mul(&fe_zinv, &fe_zinv, &(z_ratio[i]));
			/* fe_zinv = (gej_temp[i-1].z)^-1               */

			secp256k1_ge_set_gej_zinv(&ge_temp, &(gej_temp[i - 1]), &fe_zinv);
			//secp256k1_ge_to_storage(&(prec[row][i - 1]), &ge_temp);

			secp256k1_ge_storage* ROW_PRECi_1 = (secp256k1_ge_storage*)((char*)precPtr + row * precPitch) + (i - 1);
			secp256k1_ge_to_storage(ROW_PRECi_1, &ge_temp);
		}
	}
}

cudaError_t loadHashTarget(const uint32_t words[5], const uint32_t masks[5], uint32_t lenBytes, bool enabled) {
	uint32_t zero_words[5] = { 0, 0, 0, 0, 0 };
	const uint32_t* src_words = (enabled && words) ? words : zero_words;
	const uint32_t* src_masks = (enabled && masks) ? masks : zero_words;
	const uint32_t len = enabled ? lenBytes : 0u;
	const uint32_t en = enabled ? 1u : 0u;

	cudaError_t st = cudaMemcpyToSymbol(HASH_TARGET_WORDS, src_words, sizeof(uint32_t) * 5);
	if (st != cudaSuccess) return st;
	st = cudaMemcpyToSymbol(HASH_TARGET_MASKS, src_masks, sizeof(uint32_t) * 5);
	if (st != cudaSuccess) return st;
	st = cudaMemcpyToSymbol(HASH_TARGET_LEN, &len, sizeof(uint32_t));
	if (st != cudaSuccess) return st;
	return cudaMemcpyToSymbol(HASH_TARGET_ENABLED, &en, sizeof(uint32_t));
}

cudaError_t cudaMemcpyToSymbol_BLOOM_FILTER(uint8_t* _bloomFilterPtr, int count) {
	int hostValue = count + 1;
	cudaError_t err1 = cudaMemcpyToSymbol(_bloom_count, &hostValue, sizeof(int));

	cudaError_t err2 = cudaMemcpyToSymbol(_BLOOM_FILTER, &_bloomFilterPtr, sizeof(uint8_t*), count * sizeof(uint8_t*), cudaMemcpyHostToDevice);

	if (err1 != cudaSuccess) {
		return err1;
	}
	return err2;
}

__global__ void setFilterType(bool bloomUse, bool xorFilter, bool xorFilterUn, bool xorFilterUc, bool xorFilterHc)
{
	useBloom_d = bloomUse;
	useXor_d = xorFilter;
	useXorUn_d = xorFilterUn;
	useXorUc_d = xorFilterUc;
	useXorHc_d = xorFilterHc;
	uint64_t rng_counter = 0x726b2b9d438b9d4d;
	Seed = rng_splitmix64(&rng_counter);

}

__global__ void cudaXORCopy(int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {

	size_d[count] = size_h;
	arrayLength_d[count] = arrayLength_h;
	segmentCount_d[count] = segmentCount_h;
	segmentCountLength_d[count] = segmentCountLength_h;
	segmentLength_d[count] = segmentLength_h;
	segmentLengthMask_d[count] = segmentLengthMask_h;

}

__global__ void cudaXORUnCopy(int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {
	(void)segmentCountLength_h;
	(void)segmentLength_h;
	(void)segmentLengthMask_h;

	size_d_Un[count] = size_h;
	arrayLength_d_Un[count] = arrayLength_h;
	segmentCount_d_Un[count] = segmentCount_h;

}

__global__ void cudaXORUcCopy(int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {

	size_d_Uc[count] = size_h;
	arrayLength_d_Uc[count] = arrayLength_h;
	segmentCount_d_Uc[count] = segmentCount_h;
	segmentCountLength_d_Uc[count] = segmentCountLength_h;
	segmentLength_d_Uc[count] = segmentLength_h;
	segmentLengthMask_d_Uc[count] = segmentLengthMask_h;

}

__global__ void cudaXORHcCopy(int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {

	size_d_Hc[count] = size_h;
	arrayLength_d_Hc[count] = arrayLength_h;
	segmentCount_d_Hc[count] = segmentCount_h;
	segmentCountLength_d_Hc[count] = segmentCountLength_h;
	segmentLength_d_Hc[count] = segmentLength_h;
	segmentLengthMask_d_Hc[count] = segmentLengthMask_h;

}

cudaError_t cudaMemcpyToSymbol_XOR(uint32_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {
	int hostValue = count + 1;
	cudaError_t err1 = cudaMemcpyToSymbol(_xor_count, &hostValue, sizeof(int));

	cudaError_t err2 = cudaMemcpyToSymbol(fingerprints_d, &deviceFilter, sizeof(uint32_t*), count * sizeof(uint32_t*), cudaMemcpyHostToDevice);


	cudaXORCopy << <1, 1 >> > (count, size_h, arrayLength_h, segmentCount_h, segmentCountLength_h, segmentLength_h, segmentLengthMask_h);

	if (err1 != cudaSuccess) {
		return err1;
	}


	return err2;
}

cudaError_t cudaMemcpyToSymbol_XORUn(uint32_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {
	int hostValue = count + 1;
	cudaError_t err1 = cudaMemcpyToSymbol(_xor_un_count, &hostValue, sizeof(int));

	cudaError_t err2 = cudaMemcpyToSymbol(fingerprints_d_Un, &deviceFilter, sizeof(uint32_t*), count * sizeof(uint32_t*), cudaMemcpyHostToDevice);
	cudaError_t err3 = cudaMemcpyToSymbol(segmentCountLength_d_Un, &segmentCountLength_h, sizeof(size_t), count * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaError_t err4 = cudaMemcpyToSymbol(segmentLength_d_Un, &segmentLength_h, sizeof(size_t), count * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaError_t err5 = cudaMemcpyToSymbol(segmentLengthMask_d_Un, &segmentLengthMask_h, sizeof(size_t), count * sizeof(size_t), cudaMemcpyHostToDevice);


	cudaXORUnCopy << <1, 1 >> > (count, size_h, arrayLength_h, segmentCount_h, segmentCountLength_h, segmentLength_h, segmentLengthMask_h);

	if (err1 != cudaSuccess) {
		return err1;
	}
	if (err2 != cudaSuccess) return err2;
	if (err3 != cudaSuccess) return err3;
	if (err4 != cudaSuccess) return err4;
	if (err5 != cudaSuccess) return err5;


	return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol_XORUc(uint16_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {
	int hostValue = count + 1;
	cudaError_t err1 = cudaMemcpyToSymbol(_xor_uc_count, &hostValue, sizeof(int));

	cudaError_t err2 = cudaMemcpyToSymbol(fingerprints_d_Uc, &deviceFilter, sizeof(uint16_t*), count * sizeof(uint16_t*), cudaMemcpyHostToDevice);


	cudaXORUcCopy << <1, 1 >> > (count, size_h, arrayLength_h, segmentCount_h, segmentCountLength_h, segmentLength_h, segmentLengthMask_h);

	if (err1 != cudaSuccess) {
		return err1;
	}


	return err2;
}

cudaError_t cudaMemcpyToSymbol_XORHc(uint8_t* deviceFilter, int count, size_t size_h, size_t arrayLength_h, size_t segmentCount_h, size_t segmentCountLength_h, size_t segmentLength_h, size_t segmentLengthMask_h) {
	int hostValue = count + 1;
	cudaError_t err1 = cudaMemcpyToSymbol(_xor_hc_count, &hostValue, sizeof(int));

	cudaError_t err2 = cudaMemcpyToSymbol(fingerprints_d_Hc, &deviceFilter, sizeof(uint8_t*), count * sizeof(uint8_t*), cudaMemcpyHostToDevice);


	cudaXORHcCopy << <1, 1 >> > (count, size_h, arrayLength_h, segmentCount_h, segmentCountLength_h, segmentLength_h, segmentLengthMask_h);

	if (err1 != cudaSuccess) {
		return err1;
	}


	return err2;
}
cudaError_t loadWindow(unsigned int windowSize, unsigned int windows) {
	int _l[1];
	_l[0] = windows;
	cudaMemcpyToSymbol(WINDOWS_SIZE_CONST, _l, 1 * sizeof(unsigned int));
	_l[0] = windowSize;
	return cudaMemcpyToSymbol(ECMULT_WINDOW_SIZE_CONST, _l, 1 * sizeof(unsigned int));
}

// Device helper: lltoa.
__device__ int lltoa(uint64_t* __restrict__ val, char* __restrict__ buf, const char* __restrict__ dict) {
	int i = 63;
	for (; *val && i; --i, *val /= ALPHABET_LEN) {
		buf[i] = dict[*val % ALPHABET_LEN];
	}
	return i + 1;
}
