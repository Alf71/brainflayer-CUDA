#include "Kernel.cuh"

#include "lib/secp256k1/secp256k1.cuh"
#include "lib/secp256k1/secp256k1_batch_impl.cuh"
#include "lib/secp256k1/GPUWalk.cuh"
#include "sr25519-donna-32bit/sr25519.h"
#include "sr25519-donna-32bit/ed25519-donna/ed25519.h"
#include "big_int/big_int_device.cuh"
#include "lib/secp256k1/secp256k1_scalar.cuh"
#include "lib/secp256k1/secp256k1_group.cuh"
#include "lib/secp256k1/secp256k1_field.cuh"

// Advances global P0 by `advance * H_global` for persistent secp256k1 walking.
__global__ void advance_P0_kernel(uint64_t advance) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	__align__(16) secp256k1_gej Hj;
	secp256k1_gej_set_ge(&Hj, &H_global);


	__align__(16) secp256k1_gej tmp;
	secp256k1_gej_mul_u64_gej(&tmp, &Hj, advance);


	__align__(16) secp256k1_gej P0j;
	secp256k1_gej_set_ge(&P0j, &P0_global);
	secp256k1_gej_add_var(&P0j, &P0j, &tmp, NULL);


	__align__(16) secp256k1_ge P0new;
	secp256k1_ge_set_gej(&P0new, &P0j);


	P0_global = P0new;

}


// Computes initial P0 (from start key) and H (from step) in Jacobian form and stores them globally.
__global__ void compute_P0_H_kernel(
	const uint8_t* __restrict__ start_point,
	uint64_t step,
	const secp256k1_ge_storage* __restrict__ precPtr,
	size_t precPitch,
	int mode)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	__align__(16) secp256k1_scalar s_k0, s_step;
	__align__(16) secp256k1_gej gej_P0, gej_H;

	int windowLimit = WINDOWS_SIZE_CONST[0];
	int windowMultLimit = ECMULT_WINDOW_SIZE_CONST[0];

	// k0
	secp256k1_scalar_set_b32(&s_k0, start_point, NULL);
	secp256k1_ecmult_big(&gej_P0, &s_k0, precPtr, precPitch, windowLimit, windowMultLimit);

	// step > scalar 
	__align__(16) uint8_t step_be[32] = { 0 };
#pragma unroll
	for (int i = 0; i < 8; ++i) {
		step_be[31 - i] = (uint8_t)((step >> (8 * i)) & 0xFF);
	}
	secp256k1_scalar_set_b32(&s_step, step_be, NULL);
	secp256k1_ecmult_big(&gej_H, &s_step, precPtr, precPitch, windowLimit, windowMultLimit);

	__align__(16) secp256k1_ge ge_P0;
	__align__(16) secp256k1_ge ge_H;

	secp256k1_ge_set_gej(&ge_P0, &gej_P0);
	secp256k1_ge_set_gej(&ge_H, &gej_H);

	if (mode < 0) {
		secp256k1_ge_neg(&ge_H, &ge_H);
	}

	P0_global = ge_P0;
	H_global = ge_H;
}

#define VS_GRP_TABLE_SIZE 512
// Kernel: builds the i*step*G table for vanity mode (Gx, Gy, _2Gnx, _2Gny).
__global__ void vanity_set_step_table_kernel(
	uint64_t step,
	const secp256k1_ge_storage* __restrict__ precPtr,
	size_t precPitch,
	uint64_t* __restrict__ gx_out,
	uint64_t* __restrict__ gy_out,
	uint64_t* __restrict__ _2gnx_out,
	uint64_t* __restrict__ _2gny_out)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	__align__(16) uint8_t step_be[32] = { 0 };
	for (int i = 0; i < 8; i++) step_be[31 - i] = (uint8_t)((step >> (8 * i)) & 0xFF);

	__align__(16) secp256k1_scalar s_step;
	secp256k1_scalar_set_b32(&s_step, step_be, NULL);

	int wl = WINDOWS_SIZE_CONST[0];
	int wml = ECMULT_WINDOW_SIZE_CONST[0];
	__align__(16) secp256k1_gej Hj;
	secp256k1_ecmult_big(&Hj, &s_step, precPtr, precPitch, wl, wml);

	__align__(16) secp256k1_ge H;
	secp256k1_ge_set_gej(&H, &Hj);

	__align__(16) secp256k1_gej Pj;
	secp256k1_gej_set_ge(&Pj, &H);
	__align__(16) secp256k1_ge ge_tmp;

	for (int i = 0; i < VS_GRP_TABLE_SIZE; i++) {
		secp256k1_ge_set_gej(&ge_tmp, &Pj);
		__align__(16) uint64_t px[4], py[4];
		fe_to_u64x4(px, &ge_tmp.x);
		fe_to_u64x4(py, &ge_tmp.y);
		for (int w = 0; w < 4; w++) gx_out[i * 4 + w] = px[w];
		for (int w = 0; w < 4; w++) gy_out[i * 4 + w] = py[w];
		if (i < VS_GRP_TABLE_SIZE - 1)
			secp256k1_gej_add_ge(&Pj, &Pj, &H);
	}

	// _2Gn = 2 * (512*step*G) = 1024*step*G
	secp256k1_gej_double_var(&Pj, &Pj, NULL);
	secp256k1_ge_set_gej(&ge_tmp, &Pj);
	uint64_t _2gnx[4], _2gny[4];
	fe_to_u64x4(_2gnx, &ge_tmp.x);
	fe_to_u64x4(_2gny, &ge_tmp.y);
	for (int w = 0; w < 4; w++) _2gnx_out[w] = _2gnx[w];
	for (int w = 0; w < 4; w++) _2gny_out[w] = _2gny[w];
}

// Host helper: updates GPUGroup (Gx, Gy, _2Gnx, _2Gny) for step*G.
// Precompute start points P0 + (thread_steps_pub * tIx) * H into uint64[4] for each thread.
// Layout for Load256A: block b stores at [b*4*blockDim.x + w*blockDim.x + threadIdx.x]
__global__ void precompute_vanity_starts_kernel(
	uint64_t* __restrict__ startx_buf,
	uint64_t* __restrict__ starty_buf,
	int thread_steps_pub)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;

	secp256k1_gej Pstart;
	secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
	secp256k1_ge ge_start;
	secp256k1_ge_set_gej(&ge_start, &Pstart);
	uint64_t sx[4], sy[4];
	fe_to_u64x4(sx, &ge_start.x);
	fe_to_u64x4(sy, &ge_start.y);

	uint64_t* base_x = startx_buf + blockIdx.x * 4 * blockDim.x;
	uint64_t* base_y = starty_buf + blockIdx.x * 4 * blockDim.x;
	Store256A(base_x, sx);
	Store256A(base_y, sy);
}

__constant__ __align__(16) uint64_t VANITY_PERSIST_SHIFT_X[4] = { 0, 0, 0, 0 };
__constant__ __align__(16) uint64_t VANITY_PERSIST_SHIFT_Y[4] = { 0, 0, 0, 0 };

// Computes affine delta point for one persistent-iteration advance.
__global__ void vanity_compute_persistent_shift_kernel(uint64_t advance, uint64_t* out_xy)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	if (out_xy == nullptr) return;

	if (advance == 0u) {
		for (int i = 0; i < 8; ++i) out_xy[i] = 0u;
		return;
	}

	secp256k1_gej Hj, tmp;
	secp256k1_gej_set_ge(&Hj, &H_global);
	secp256k1_gej_mul_u64_gej(&tmp, &Hj, advance);

	secp256k1_ge d;
	secp256k1_ge_set_gej(&d, &tmp);
	fe_to_u64x4(out_xy + 0, &d.x);
	fe_to_u64x4(out_xy + 4, &d.y);
}

// Applies cached persistent delta to each precomputed vanity start point.
__global__ void vanity_apply_persistent_shift_kernel(
	uint64_t* __restrict__ startx_buf,
	uint64_t* __restrict__ starty_buf)
{
	if (startx_buf == nullptr || starty_buf == nullptr) return;

	uint64_t* base_x = startx_buf + blockIdx.x * 4 * blockDim.x;
	uint64_t* base_y = starty_buf + blockIdx.x * 4 * blockDim.x;
	uint64_t sx[4], sy[4];
	Load256A(sx, base_x);
	Load256A(sy, base_y);

	secp256k1_fe fx, fy, dx, dy;
	u64x4_to_fe(&fx, sx);
	u64x4_to_fe(&fy, sy);
	u64x4_to_fe(&dx, VANITY_PERSIST_SHIFT_X);
	u64x4_to_fe(&dy, VANITY_PERSIST_SHIFT_Y);

	secp256k1_ge P, D;
	P.infinity = 0;
	P.x = fx;
	P.y = fy;
	D.infinity = 0;
	D.x = dx;
	D.y = dy;

	__align__(16) secp256k1_gej Pj;
	secp256k1_gej_set_ge(&Pj, &P);
	secp256k1_gej_add_ge(&Pj, &Pj, &D);

	secp256k1_ge R;
	secp256k1_ge_set_gej(&R, &Pj);
	fe_to_u64x4(sx, &R.x);
	fe_to_u64x4(sy, &R.y);

	Store256A(base_x, sx);
	Store256A(base_y, sy);
}

// Host helper that updates constant-memory persistent delta vectors.
cudaError_t cuda_vanity_set_persistent_shift(uint64_t advance) {
	uint64_t zeros[4] = { 0, 0, 0, 0 };
	if (advance == 0u) {
		cudaError_t st0 = cudaMemcpyToSymbol(VANITY_PERSIST_SHIFT_X, zeros, sizeof(zeros));
		if (st0 != cudaSuccess) return st0;
		return cudaMemcpyToSymbol(VANITY_PERSIST_SHIFT_Y, zeros, sizeof(zeros));
	}

	uint64_t* d_xy = nullptr;
	cudaError_t st = cudaMalloc((void**)&d_xy, sizeof(uint64_t) * 8);
	if (st != cudaSuccess) return st;

	vanity_compute_persistent_shift_kernel << <1, 1 >> > (advance, d_xy);
	st = cudaGetLastError();
	if (st == cudaSuccess) st = cudaDeviceSynchronize();

	uint64_t host_xy[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	if (st == cudaSuccess) {
		st = cudaMemcpy(host_xy, d_xy, sizeof(host_xy), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_xy);
	if (st != cudaSuccess) return st;

	st = cudaMemcpyToSymbol(VANITY_PERSIST_SHIFT_X, host_xy + 0, sizeof(uint64_t) * 4);
	if (st != cudaSuccess) return st;
	return cudaMemcpyToSymbol(VANITY_PERSIST_SHIFT_Y, host_xy + 4, sizeof(uint64_t) * 4);
}

// Host helper for precomputed start buffers.
cudaError_t cuda_vanity_apply_persistent_shift(uint64_t* startx_buf, uint64_t* starty_buf, dim3 grid, dim3 block) {
	if (startx_buf == nullptr || starty_buf == nullptr) {
		return cudaErrorInvalidDevicePointer;
	}
	vanity_apply_persistent_shift_kernel << <grid, block >> > (startx_buf, starty_buf);
	return cudaGetLastError();
}

// Rebuilds Gx/Gy group table and 2G constants for arbitrary vanity `step`.
cudaError_t cuda_vanity_set_step_increment(uint64_t step, const void* precPtr, size_t precPitch) {
	if (step == 1) return cudaSuccess;  // Default table is already valid for step=1.
	if (precPtr == nullptr) return cudaErrorInvalidValue;

	const size_t table_words = VS_GRP_TABLE_SIZE * 4;
	const size_t total_words = table_words * 2 + 8;
	uint64_t* d_gx = nullptr;
	cudaError_t err = cudaMalloc(&d_gx, total_words * sizeof(uint64_t));
	if (err != cudaSuccess) return err;
	uint64_t* d_gy = d_gx + table_words;
	uint64_t* d_2gnx = d_gy + table_words;
	uint64_t* d_2gny = d_2gnx + 4;

	vanity_set_step_table_kernel << <1, 1 >> > (
		step, (const secp256k1_ge_storage*)precPtr, precPitch,
		d_gx, d_gy, d_2gnx, d_2gny);
	err = cudaGetLastError();
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }

	uint64_t host_gx[VS_GRP_TABLE_SIZE][4];
	uint64_t host_gy[VS_GRP_TABLE_SIZE][4];
	uint64_t host_2gnx[4], host_2gny[4];
	err = cudaMemcpy(host_gx, d_gx, sizeof(host_gx), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }
	err = cudaMemcpy(host_gy, d_gy, sizeof(host_gy), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }
	err = cudaMemcpy(host_2gnx, d_2gnx, sizeof(host_2gnx), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }
	err = cudaMemcpy(host_2gny, d_2gny, sizeof(host_2gny), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cudaFree(d_gx); return err; }
	cudaFree(d_gx);

	err = cudaMemcpyToSymbol(Gx, host_gx, sizeof(host_gx));
	if (err != cudaSuccess) return err;
	err = cudaMemcpyToSymbol(Gy, host_gy, sizeof(host_gy));
	if (err != cudaSuccess) return err;
	err = cudaMemcpyToSymbol(_2Gnx, host_2gnx, sizeof(host_2gnx));
	if (err != cudaSuccess) return err;
	return cudaMemcpyToSymbol(_2Gny, host_2gny, sizeof(host_2gny));
}

// Single maintained PRIV sequential kernel.

// Unified private-key sequential worker for mixed secp256k1/ed25519 paths.
__global__ void workerPRIV_seq_128(
	bool* isResult,
	bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr,
	const size_t precPitch,
	int mode,
	uint8_t* start_point,
	uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf,
	uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;

	uint64_t starter = thread_steps_pub * (uint64_t)tIx;


	const bool secp256_dev = secp256_d;
	const bool ed25519_dev = ed25519_d;
	const bool compressed = compressed_dev;
	const bool uncompressed = uncompressed_dev;
	const bool segwit = segwit_dev;
	const bool taproot = taproot_dev;
	const bool ethereum = ethereum_dev;
	const bool xpoint = xpoint_dev;
	const bool solana = solana_dev;
	const bool ton = ton_dev;
	const bool ton_all = ton_all_dev;
	const bool dot = dot_dev;
	const bool aptos = aptos_dev;
	const bool sui = sui_dev;
	const bool xrp = xrp_dev;
	const bool iota = iota_dev;
	const bool icp = icp_dev;
	const bool fil = fil_dev;
	const bool xtz = xtz_dev;

	__align__(16) uint32_t hash160[8];
	unsigned char pubKeys[65];
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];

	if (secp256_dev) {
		bool use_vanity = (thread_steps_pub % VS_GRP_SIZE == 0);
		const bool endomorphism = endomorphism_dev;

		if (use_vanity) {
			uint64_t sx[4], sy[4];
			if (startx_buf != nullptr && starty_buf != nullptr) {
				uint64_t* base_x = startx_buf + blockIdx.x * 4 * blockDim.x;
				uint64_t* base_y = starty_buf + blockIdx.x * 4 * blockDim.x;
				Load256A(sx, base_x);
				Load256A(sy, base_y);
			} else {
				secp256k1_gej Pstart;
				secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
				secp256k1_ge ge_start;
				secp256k1_ge_set_gej(&ge_start, &Pstart);
				fe_to_u64x4(sx, &ge_start.x);
				fe_to_u64x4(sy, &ge_start.y);
			}

			bool need_full_y = uncompressed || ethereum || taproot || sui || iota || aptos || icp || fil || xtz;
			int numBatches = thread_steps_pub / VS_GRP_SIZE;
			for (int batch = 0; batch < numBatches; batch++) {
				struct ProcessPointFn {
					bool* buffResult; bool* isResult;
					const secp256k1_ge_storage* precPtr; size_t precPitch;
					int mode; uint64_t step; uint64_t starter;
					uint32_t* c; uint8_t* priv_bytes; uint32_t* hash160; unsigned char* pubKeys;
					int batch;
					bool uncompressed, compressed, segwit, ethereum, taproot, xpoint, xrp, sui, iota, aptos, icp, fil, xtz, endomorphism;
// Device helper: operator.
					__device__ __forceinline__ void operator()(uint64_t* px, uint64_t* py, int pkField) const {
						uint64_t batchBase = starter + (uint64_t)(batch * VS_GRP_SIZE);
						uint64_t idx = (mode == 1) ? (batchBase + (uint64_t)pkField) : (batchBase + (uint64_t)VS_GRP_SIZE - (uint64_t)pkField);
						uint8_t odd_py = (uint8_t)(py[0] & 1);
						bool hit = false;
						#define _CP_IN() do { if (hit) break; uint32_t r[8]; uint64_t off_lo, off_hi; mul_u64_u64_to_u128(idx, step, off_lo, off_hi); if (mode == 1) add128to256_device(c, off_lo, off_hi, r); else sub128from256_device(c, off_lo, off_hi, r); storeWordsToBE256(r, priv_bytes); hit = true; } while(0)

						if (uncompressed) {
						_GetHash160Uncomp_fast(px, py, (uint8_t*)hash160);
						if (checkHash(hash160)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x01;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash160, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (compressed) {
						if (endomorphism) {
							// Evaluate base, beta*x and beta^2*x compressed variants with parity-aware endomorphism tags.
							uint64_t pe1x[4], pe2x[4];
							uint32_t hash_odd[8], hash_even[8];
							_ModMult(pe1x, px, _beta);
							_ModMult(pe2x, px, _beta2);
							for (int j = 0; j < 3; ++j) {
								uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
								uint8_t vbase = (uint8_t)(j * 2);
								uint8_t odd_variant = (uint8_t)(vbase + (odd_py ? ENDO_VARIANT_BASE_POS : ENDO_VARIANT_BASE_NEG));
								uint8_t even_variant = (uint8_t)(vbase + (odd_py ? ENDO_VARIANT_BASE_NEG : ENDO_VARIANT_BASE_POS));
								_GetHash160CompSym(xcur, (uint8_t*)hash_odd, (uint8_t*)hash_even);
								if (checkHash(hash_odd)) {
									_CP_IN();
									buffResult[0] = true;
									isResult[0] = true;
									uint8_t type = (uint8_t)(ENDO_TAG_BASE + ENDO_GROUP_STRIDE * ENDO_GROUP_COMPRESSED + odd_variant);
									unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
									if (ridx < MAX_FOUNDS_DEV) {
										memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
										memcpy(d_foundHash160[ridx], &hash_odd, 32);
										d_type[ridx] = type;
									}
								}
								if (checkHash(hash_even)) {
									_CP_IN();
									buffResult[0] = true;
									isResult[0] = true;
									uint8_t type = (uint8_t)(ENDO_TAG_BASE + ENDO_GROUP_STRIDE * ENDO_GROUP_COMPRESSED + even_variant);
									unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
									if (ridx < MAX_FOUNDS_DEV) {
										memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
										memcpy(d_foundHash160[ridx], &hash_even, 32);
										d_type[ridx] = type;
									}
								}
							}
						} else {
							_GetHash160Comp_fast(px, odd_py, (uint8_t*)hash160);
							if (checkHash(hash160)) {
								_CP_IN();
								buffResult[0] = true;
								isResult[0] = true;
								uint8_t type = 0x02;
								unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
								if (ridx < MAX_FOUNDS_DEV) {
									memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
									memcpy(d_foundHash160[ridx], &hash160, 32);
									d_type[ridx] = type;
								}
							}
						}
					}
					if (segwit) {
						if (!compressed) _GetHash160Comp_fast(px, odd_py, (uint8_t*)hash160);
						_GetHash160P2SHCompFromHash(hash160, hash160);
						if (checkHash(hash160)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x03;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash160, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (ethereum) {
						u64x4_to_pubkey65(px, py, pubKeys);
						unsigned char keccak_hash[32];
						keccak((char*)&pubKeys[1], 64, keccak_hash, 32);
						__align__(16) uint32_t eth_hash160[8] = { 0 };
						for (int h = 12, i = 0; i < 5; i++) {
							eth_hash160[i] = (keccak_hash[h++]) | ((keccak_hash[h++] << 8) & 0x0000ff00) | ((keccak_hash[h++] << 16) & 0x00ff0000) | ((keccak_hash[h++] << 24) & 0xff000000);
						}
						if (checkHash(eth_hash160)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x06;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], eth_hash160, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (taproot) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t TaprootHash[32];
						TweakTaproot(&TaprootHash[0], &pubKeys[0], precPtr, precPitch);
						_GetRMD160((uint32_t*)TaprootHash, hash160);
						if (checkHash(hash160)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x04;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], TaprootHash, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (xpoint) {
						uint32_t xpoint_hash[8];
						u64x4_to_x32(px, (unsigned char*)xpoint_hash);
						if (checkHash(xpoint_hash)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x05;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &xpoint_hash, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (xrp) {
						_GetHash160Comp_fast(px, odd_py, (uint8_t*)hash160);
						if (checkHash(hash160)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x90;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash160, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (sui) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t buff[34];
						memcpy(&buff[2], &pubKeys[1], 32);
						buff[0] = 0x01;
						buff[1] = 0x2 + (pubKeys[64] & 1);
						uint8_t hash_addr[32];
						Blake2b_256(buff, 34, hash_addr);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x70;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash_addr, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (iota) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t buff[34];
						memcpy(&buff[2], &pubKeys[1], 32);
						buff[0] = 0x01;
						buff[1] = 0x2 + (pubKeys[64] & 1);
						uint8_t hash_addr[32];
						Blake2b_256(buff, 34, hash_addr);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x50;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash_addr, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (aptos) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t buff[35] = { 0x00 };
						memcpy(&buff[2], &pubKeys[1], 32);
						buff[0] = 0x01;
						buff[1] = 0x2 + (pubKeys[64] & 1);
						buff[34] = 0x02;
						uint8_t hash_addr[32];
						sha3_256((char*)buff, 35, hash_addr);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x22;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], hash_addr, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (icp) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t principal[29] = { 0 };
						icp_principal_from_secp256k1(&pubKeys[0], &principal[0]);
						uint8_t hash_addr[32];
						icp_account_identifier(&principal[0], NULL, &hash_addr[0]);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x53;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash_addr, 32);
								d_type[ridx] = type;
							}
						}
					}
					if (fil) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t hash_addr[20];
						Blake2b_160(&pubKeys[0], 65, &hash_addr[0]);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x41;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash_addr, 20);
								d_type[ridx] = type;
							}
						}
						unsigned char keccak_hash[32];
						keccak((char*)&pubKeys[1], 64, keccak_hash, 32);
						uint32_t hash160_fil[8];
						for (int h = 12, i = 0; i < 5; i++) {
							hash160_fil[i] = (keccak_hash[h++]) | ((keccak_hash[h++] << 8) & 0x0000ff00) | ((keccak_hash[h++] << 16) & 0x00ff0000) | ((keccak_hash[h++] << 24) & 0xff000000);
						}
						if (checkHash(hash160_fil)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x42;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash160_fil, 20);
								d_type[ridx] = type;
							}
						}
					}
					if (xtz) {
						u64x4_to_pubkey65(px, py, pubKeys);
						uint8_t buff[33] = { 0x00 };
						memcpy(&buff[1], &pubKeys[1], 32);
						buff[0] = 0x2 + (pubKeys[64] & 1);
						uint8_t hash_addr[20];
						Blake2b_160(&buff[0], 33, &hash_addr[0]);
						if (checkHash((uint32_t*)hash_addr)) {
							_CP_IN();
							buffResult[0] = true;
							isResult[0] = true;
							uint8_t type = 0x92;
							unsigned long long int ridx = atomicAdd(&d_resultsCount[0], 1);
							if (ridx < MAX_FOUNDS_DEV) {
								memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
								memcpy(d_foundHash160[ridx], &hash_addr, 20);
								d_type[ridx] = type;
							}
						}
					}
						#undef _CP_IN
					}
				};
				ProcessPointFn process_point = { buffResult, isResult, precPtr, precPitch, mode, step, starter, c, priv_bytes, hash160, pubKeys, batch, uncompressed, compressed, segwit, ethereum, taproot, xpoint, xrp, sui, iota, aptos, icp, fil, xtz, endomorphism };
				if (need_full_y) {
					vanity_walk_batch_1024_impl<true>(sx, sy, process_point);
				} else {
					vanity_walk_batch_1024_impl<false>(sx, sy, process_point);
				}
			}
		}
	}

	if (ed25519_dev)
	{
		for (int pkField = 0; pkField < thread_steps_pub; pkField++) {
				uint32_t r[8];
				uint64_t idx = starter + (uint64_t)pkField;
				uint64_t off_lo, off_hi;
				mul_u64_u64_to_u128(idx, step, off_lo, off_hi);

				if (mode == 1) {
					add128to256_device(c, off_lo, off_hi, r);
				}
				else {
					sub128from256_device(c, off_lo, off_hi, r);
				}
				storeWordsToBE256(r, priv_bytes);


			unsigned char publ[32];
			unsigned char pkey[32];
			memcpy(pkey, &priv_bytes[0], 32);

			ed25519_key_to_pub(pkey, publ);

			if (solana)
			{
				if (checkHash((uint32_t*)publ)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x60;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &publ[0], 32);
						d_type[idx] = type;
					}
				}
			}
			if (dot)
			{
				unsigned char public_key_sr[32];
				sr25519_keypair keypair = { 0 };
				sr25519_keypair_from_seed(keypair, pkey);
				memcpy(public_key_sr, keypair + 64, 32);
				if (checkHash((uint32_t*)public_key_sr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x31;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &public_key_sr, 32);
						d_type[idx] = type;
					}
				}
				if (checkHash((uint32_t*)publ)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x30;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &publ[0], 32);
						d_type[idx] = type;
					}
				}

			}
			if (ton)
			{
				uint8_t v3r1[32];
				pubkey_to_hash_ton(publ, "v3r1", v3r1);
				if (checkHash((uint32_t*)v3r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x85;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v3r1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v3r2[32];
				pubkey_to_hash_ton(publ, "v3r2", v3r2);

				if (checkHash((uint32_t*)v3r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x86;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v3r2, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v4r2[32];
				pubkey_to_hash_ton(publ, "v4r2", v4r2);
				if (checkHash((uint32_t*)v4r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x88;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v4r2, 32);
						d_type[idx] = type;
					}
				}
				uint8_t v5r1[32];
				pubkey_to_hash_ton(publ, "v5r1", v5r1);
				if (checkHash((uint32_t*)v5r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x89;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {
						memcpy(d_foundPrvKeys[idx], &pkey, 32);
						memcpy(d_foundHash160[idx], &v5r1, 32);
						d_type[idx] = type;
					}
				}
				uint8_t hv3[32];
				pubkey_to_hash_ton(publ, "hv3", hv3);

				if (checkHash((uint32_t*)hv3)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x8c;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hv3, 32);
						d_type[idx] = type;
					}
				}
			}
			if (ton_all)
			{
				uint8_t v1r1[32];
				pubkey_to_hash_ton(publ, "v1r1", v1r1);
				if (checkHash((uint32_t*)v1r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x80;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v1r1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v1r2[32];
				pubkey_to_hash_ton(publ, "v1r2", v1r2);
				if (checkHash((uint32_t*)v1r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x81;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v1r2, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v1r3[32];
				pubkey_to_hash_ton(publ, "v1r3", v1r3);
				if (checkHash((uint32_t*)v1r3)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x82;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v1r3, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v2r1[32];
				pubkey_to_hash_ton(publ, "v2r1", v2r1);
				if (checkHash((uint32_t*)v2r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x83;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v2r1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v2r2[32];
				pubkey_to_hash_ton(publ, "v2r2", v2r2);
				if (checkHash((uint32_t*)v2r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x84;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v2r2, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v3r1[32];
				pubkey_to_hash_ton(publ, "v3r1", v3r1);
				if (checkHash((uint32_t*)v3r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x85;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v3r1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v3r2[32];
				pubkey_to_hash_ton(publ, "v3r2", v3r2);
				if (checkHash((uint32_t*)v3r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x86;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v3r2, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v4r1[32];
				pubkey_to_hash_ton(publ, "v4r1", v4r1);
				if (checkHash((uint32_t*)v4r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x87;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v4r1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t v4r2[32];
				pubkey_to_hash_ton(publ, "v4r2", v4r2);
				if (checkHash((uint32_t*)v4r2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x88;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &v4r2, 32);
						d_type[idx] = type;
					}
				}
				uint8_t v5r1[32];
				pubkey_to_hash_ton(publ, "v5r1", v5r1);
				if (checkHash((uint32_t*)v5r1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x89;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {
						memcpy(d_foundPrvKeys[idx], &pkey, 32);
						memcpy(d_foundHash160[idx], &v5r1, 32);
						d_type[idx] = type;
					}
				}
				uint8_t hv1[32];
				pubkey_to_hash_ton(publ, "hv1", hv1);
				if (checkHash((uint32_t*)hv1)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x8a;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hv1, 32);
						d_type[idx] = type;
					}
				}

				uint8_t hv2[32];
				pubkey_to_hash_ton(publ, "hv2", hv2);
				if (checkHash((uint32_t*)hv2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x8b;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hv2, 32);
						d_type[idx] = type;
					}
				}

				uint8_t hv3[32];
				pubkey_to_hash_ton(publ, "hv3", hv3);
				if (checkHash((uint32_t*)hv3)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x8c;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hv3, 32);
						d_type[idx] = type;
					}
				}
			}
			if (xrp) {
				int keyLenSkip = 0;
				_GetHash160ED(publ, keyLenSkip, (uint8_t*)hash160);
				if (checkHash(hash160)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x91;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {
						memcpy(d_foundPrvKeys[idx], &pkey, 32);
						memcpy(d_foundHash160[idx], &hash160, 32);
						d_type[idx] = type;
					}
				}


			}
			if (aptos)
			{
				uint8_t buff[33] = { 0x00 };
				uint8_t buff2[34] = { 0x00 };
				memcpy(buff, publ, 32);
				memcpy(&buff2[1], publ, 32);
				buff2[33] = 0x02;
				uint8_t hash_addr[32];
				uint8_t hash_addr2[32];
				sha3_256((char*)buff, 33, hash_addr);
				sha3_256((char*)buff2, 34, hash_addr2);
				if (checkHash((uint32_t*)hash_addr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x20;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr, 32);
						d_type[idx] = type;
					}
				}

				if (checkHash((uint32_t*)hash_addr2)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x21;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr2, 32);
						d_type[idx] = type;
					}
				}


			}
			if (sui)
			{
				uint8_t buff[33] = { 0x00 };
				memcpy(&buff[1], publ, 32);
				uint8_t hash_addr[32];
				Blake2b_256(buff, 33, hash_addr);
				if (checkHash((uint32_t*)hash_addr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x71;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr, 32);
						d_type[idx] = type;
					}
				}
			}
			if (iota)
			{
				uint8_t hash_addr[32];
				Blake2b_256(publ, 32, hash_addr);
				if (checkHash((uint32_t*)hash_addr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x51;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {

						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr, 32);
						d_type[idx] = type;
					}
				}
			}
			if (icp)
			{

				uint8_t principal[29] = { 0 };
				icp_principal_from_ed25519(&publ[0], &principal[0]);
				uint8_t hash_addr[32];
				icp_account_identifier(&principal[0], NULL, &hash_addr[0]);
				if (checkHash((uint32_t*)hash_addr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x52;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr, 32);
						d_type[idx] = type;
					}
				}

			}
			if (xtz)
			{
				uint8_t hash_addr[20];
				Blake2b_160(&publ[0], 32, &hash_addr[0]);

				if (checkHash((uint32_t*)hash_addr)) {
					buffResult[0] = true;
					isResult[0] = true;
					uint8_t type = 0x93;
					unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
					if (idx < MAX_FOUNDS_DEV) {


						memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
						memcpy(d_foundHash160[idx], &hash_addr, 20);
						d_type[idx] = type;
					}
				}
			}
		}

	}
}
