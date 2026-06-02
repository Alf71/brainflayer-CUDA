// Vanity kernels - separate .cu module, no templates
#include "Kernel.cuh"
#include "lib/secp256k1/secp256k1.cuh"
#include "lib/secp256k1/secp256k1_batch_impl.cuh"
#include "lib/secp256k1/GPUWalk.cuh"
#include "big_int/big_int_device.cuh"
#include "lib/secp256k1/secp256k1_scalar.cuh"
#include "lib/secp256k1/secp256k1_group.cuh"
#include "lib/secp256k1/secp256k1_field.cuh"

// Implementations (declared in GPUWalk.cuh) - single TU to avoid multiple definition
__device__ void vanity_walk_1024_parity(uint64_t sx[4], uint64_t sy[4], vanity_process_point_fn fn, void* ctx, int batch)
{
	uint64_t dx[4], px[4], py[4], py_full[4], dy[4];
	uint64_t sxn[4], syn[4], sx_gx[4];
	uint8_t odd_py;
	uint64_t inverse[5];
	uint64_t subp[VS_GRP_HALF][4];
	Load256(py_full, sy);
	fn(sx, py_full, VS_GRP_HALF, batch, ctx);
	ModSub256(sxn, _2Gnx, sx);
	Load256(subp[VS_GRP_HALF - 1], sxn);
	for (int i = VS_GRP_HALF - 1; i > 0; i--) {
		ModSub256(syn, Gx[i], sx);
		_ModMult(sxn, syn);
		Load256(subp[i - 1], sxn);
	}
	ModSub256(inverse, Gx[0], sx);
	_ModMult(inverse, sxn);
	inverse[4] = 0;
	_ModInv(inverse);
	ModNeg256(syn, sy);
	ModNeg256(sxn, sx);
	uint32_t i;
	for (i = 0; i < VS_GRP_HALF - 1; i++) {
		ModSub256(sx_gx, Gx[i], sxn);
		_ModMult(dx, subp[i], inverse);
		ModSub256(dy, Gy[i], sy);
		_ModMult(dy, dx);
		_ModSqr(px, dy);
		ModSub256(px, sx_gx);
		ModSub256(py, sx, px);
		_ModMult(py, dy);
		ModSub256isOdd(py, sy, &odd_py);
		py_full[0] = (py_full[0] & ~1ULL) | odd_py;
		fn(px, py_full, VS_GRP_HALF + (i + 1), batch, ctx);
		ModSub256(dy, syn, Gy[i]);
		_ModMult(dy, dx);
		_ModSqr(px, dy);
		ModSub256(px, sx_gx);
		ModSub256(py, px, sx);
		_ModMult(py, dy);
		ModSub256isOdd(syn, py, &odd_py);
		py_full[0] = (py_full[0] & ~1ULL) | odd_py;
		fn(px, py_full, VS_GRP_HALF - (i + 1), batch, ctx);
		ModSub256(dx, Gx[i], sx);
		_ModMult(inverse, dx);
	}
	_ModMult(dx, subp[i], inverse);
	ModSub256(dy, syn, Gy[i]);
	_ModMult(dy, dx);
	_ModSqr(px, dy);
	ModSub256(px, sx);
	ModSub256(px, Gx[i]);
	ModSub256(py, px, sx);
	_ModMult(py, dy);
	ModSub256isOdd(syn, py, &odd_py);
	py_full[0] = (py_full[0] & ~1ULL) | odd_py;
	fn(px, py_full, 0, batch, ctx);
	ModSub256(dy, _2Gny, sy);
	ModSub256(dx, Gx[i], sx);
	_ModMult(inverse, dx);
	_ModMult(dy, inverse);
	_ModSqr(px, dy);
	ModSub256(px, sx);
	ModSub256(px, _2Gnx);
	ModSub256(py, _2Gnx, px);
	_ModMult(py, dy);
	ModSub256(py, _2Gny);
	Load256(sx, px);
	Load256(sy, py);
}

// Device helper: vanity_walk_1024_full_y.
__device__ void vanity_walk_1024_full_y(uint64_t sx[4], uint64_t sy[4], vanity_process_point_fn fn, void* ctx, int batch)
{
	uint64_t dx[4], px[4], py[4], py_full[4], dy[4];
	uint64_t sxn[4], syn[4], sx_gx[4];
	uint64_t inverse[5];
	uint64_t subp[VS_GRP_HALF][4];
	Load256(py_full, sy);
	fn(sx, py_full, VS_GRP_HALF, batch, ctx);
	ModSub256(sxn, _2Gnx, sx);
	Load256(subp[VS_GRP_HALF - 1], sxn);
	for (int i = VS_GRP_HALF - 1; i > 0; i--) {
		ModSub256(syn, Gx[i], sx);
		_ModMult(sxn, syn);
		Load256(subp[i - 1], sxn);
	}
	ModSub256(inverse, Gx[0], sx);
	_ModMult(inverse, sxn);
	inverse[4] = 0;
	_ModInv(inverse);
	ModNeg256(syn, sy);
	ModNeg256(sxn, sx);
	uint32_t i;
	for (i = 0; i < VS_GRP_HALF - 1; i++) {
		ModSub256(sx_gx, Gx[i], sxn);
		_ModMult(dx, subp[i], inverse);
		ModSub256(dy, Gy[i], sy);
		_ModMult(dy, dx);
		_ModSqr(px, dy);
		ModSub256(px, sx_gx);
		ModSub256(py, sx, px);
		_ModMult(py, dy);
		ModSub256(py_full, py, sy);
		fn(px, py_full, VS_GRP_HALF + (i + 1), batch, ctx);
		ModSub256(dy, syn, Gy[i]);
		_ModMult(dy, dx);
		_ModSqr(px, dy);
		ModSub256(px, sx_gx);
		ModSub256(py, px, sx);
		_ModMult(py, dy);
		ModNeg256(py_full, py);
		ModSub256(py_full, py_full, sy);
		fn(px, py_full, VS_GRP_HALF - (i + 1), batch, ctx);
		ModSub256(dx, Gx[i], sx);
		_ModMult(inverse, dx);
	}
	_ModMult(dx, subp[i], inverse);
	ModSub256(dy, syn, Gy[i]);
	_ModMult(dy, dx);
	_ModSqr(px, dy);
	ModSub256(px, sx);
	ModSub256(px, Gx[i]);
	ModSub256(py, px, sx);
	_ModMult(py, dy);
	ModNeg256(py_full, py);
	ModSub256(py_full, py_full, sy);
	fn(px, py_full, 0, batch, ctx);
	ModSub256(dy, _2Gny, sy);
	ModSub256(dx, Gx[i], sx);
	_ModMult(inverse, dx);
	_ModMult(dy, inverse);
	_ModSqr(px, dy);
	ModSub256(px, sx);
	ModSub256(px, _2Gnx);
	ModSub256(py, _2Gnx, px);
	_ModMult(py, dy);
	ModSub256(py, _2Gny);
	Load256(sx, px);
	Load256(sy, py);
}

// Device helper: vanity_filters_active.
__device__ __forceinline__ bool vanity_filters_active()
{
	return FULL_d || (HASH_TARGET_ENABLED[0] != 0u) ||
		useBloom_d || useXor_d || useXorUn_d || useXorUc_d || useXorHc_d;
}

struct VanityNoopPoint {
// Device helper: operator.
	__device__ __forceinline__ void operator()(uint64_t* px, uint64_t* py, int pkField) const
	{
		(void)px;
		(void)py;
		(void)pkField;
	}
};

// Context passed to process callbacks
struct VanityCtx {
	uint32_t* c;
	uint8_t* priv_bytes;
	int mode;
	uint64_t step;
	uint64_t starter;
	bool* isResult;
	bool* buffResult;
	uint32_t* hash160;
	unsigned char* pubKeys;
	const secp256k1_ge_storage* precPtr;
	size_t precPitch;
};

// Device helper: vanity_make_endo_type.
__device__ __forceinline__ uint8_t vanity_make_endo_type(uint8_t group, uint8_t variant)
{
	return (uint8_t)(ENDO_TAG_BASE + ENDO_GROUP_STRIDE * group + variant);
}

__device__ __forceinline__ static void vanity_save_result(uint32_t* c, uint8_t* priv_bytes, int mode, uint64_t idx, uint64_t step,
	bool* isResult, bool* buffResult, const void* hash_src, int hash_len, uint8_t type_val)
{
	uint32_t r[8]; uint64_t off_lo, off_hi;
	mul_u64_u64_to_u128(idx, step, off_lo, off_hi);
	if (mode == 1) add128to256_device(c, off_lo, off_hi, r);
	else sub128from256_device(c, off_lo, off_hi, r);
	storeWordsToBE256(r, priv_bytes);
	buffResult[0] = true; isResult[0] = true;
	unsigned long long ridx = atomicAdd(&d_resultsCount[0], 1);
	if (ridx < MAX_FOUNDS_DEV) {
		int n = hash_len;
		if (n < 0) n = 0;
		if (n > (int)(20 * sizeof(uint32_t))) n = (int)(20 * sizeof(uint32_t));
		memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
		memset(d_foundHash160[ridx], 0, 20 * sizeof(uint32_t));
		if (n > 0) memcpy(d_foundHash160[ridx], hash_src, n);
		d_type[ridx] = type_val;
	}
}

// Vanity callback for x-only matching mode.
__device__ static void process_point_x(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	uint64_t idx = ctx->starter + (uint64_t)(batch * VS_GRP_SIZE + pkField);
	uint32_t xpoint_hash[8];
	u64x4_to_x32(px, (unsigned char*)xpoint_hash);
	if (checkHash(xpoint_hash))
		vanity_save_result(ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step, ctx->isResult, ctx->buffResult, xpoint_hash, 32, 0x05);
}

// Vanity callback for uncompressed HASH160 matching mode.
__device__ static void process_point_u(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	uint64_t idx = ctx->starter + (uint64_t)(batch * VS_GRP_SIZE + pkField);
	_GetHash160Uncomp_fast(px, py, (uint8_t*)ctx->hash160);
	if (checkHash(ctx->hash160))
		vanity_save_result(ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step, ctx->isResult, ctx->buffResult, ctx->hash160, 20, 0x01);
}

// Vanity callback for SegWit(P2SH-P2WPKH) matching mode.
__device__ static void process_point_s(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	uint64_t idx = ctx->starter + (uint64_t)(batch * VS_GRP_SIZE + pkField);
	uint8_t odd_py = (uint8_t)(py[0] & 1);
	_GetHash160Comp_fast(px, odd_py, (uint8_t*)ctx->hash160);
	_GetHash160P2SHCompFromHash(ctx->hash160, ctx->hash160);
	if (checkHash(ctx->hash160))
		vanity_save_result(ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step, ctx->isResult, ctx->buffResult, ctx->hash160, 20, 0x03);
}

// Vanity callback for Taproot matching mode.
__device__ static void process_point_r(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	uint64_t idx = ctx->starter + (uint64_t)(batch * VS_GRP_SIZE + pkField);
	u64x4_to_pubkey65(px, py, ctx->pubKeys);
	uint8_t TaprootHash[32];
	TweakTaproot(&TaprootHash[0], &ctx->pubKeys[0], ctx->precPtr, ctx->precPitch);
	_GetRMD160((uint32_t*)TaprootHash, ctx->hash160);
	if (checkHash(ctx->hash160))
		vanity_save_result(ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step, ctx->isResult, ctx->buffResult, TaprootHash, 32, 0x04);
}

// Vanity callback for Ethereum address-tail matching mode.
__device__ static void process_point_e(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	uint64_t idx = ctx->starter + (uint64_t)(batch * VS_GRP_SIZE + pkField);
	uint32_t eth_hash160[5];
	keccak_eth64_tail20_from_xy(px, py, eth_hash160);
	if (checkHash(eth_hash160))
		vanity_save_result(ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step, ctx->isResult, ctx->buffResult, eth_hash160, 20, 0x06);
}

// Device helper: vanity_cusr_cp_once.
__device__ __forceinline__ static void vanity_cusr_cp_once(bool& done, uint32_t* c, uint8_t* priv_bytes, int mode, uint64_t idx, uint64_t step)
{
	if (done) return;
	uint32_t r[8]; uint64_t off_lo, off_hi;
	mul_u64_u64_to_u128(idx, step, off_lo, off_hi);
	if (mode == 1) add128to256_device(c, off_lo, off_hi, r);
	else sub128from256_device(c, off_lo, off_hi, r);
	storeWordsToBE256(r, priv_bytes);
	done = true;
}

// Device helper: vanity_emit.
__device__ __forceinline__ static void vanity_emit(bool* isResult, bool* buffResult, uint8_t* priv_bytes, const void* hash_src, int len, uint8_t t)
{
	buffResult[0] = true; isResult[0] = true;
	unsigned long long ridx = atomicAdd(&d_resultsCount[0], 1);
	if (ridx < MAX_FOUNDS_DEV) {
		int n = len;
		if (n < 0) n = 0;
		if (n > (int)(20 * sizeof(uint32_t))) n = (int)(20 * sizeof(uint32_t));
		memcpy(d_foundPrvKeys[ridx], priv_bytes, 32);
		memset(d_foundHash160[ridx], 0, 20 * sizeof(uint32_t));
		if (n > 0) memcpy(d_foundHash160[ridx], hash_src, n);
		d_type[ridx] = t;
	}
}

// Unified vanity point handler for combined secp256k1 modes (c/u/s/r/e/x) with optional endomorphism fan-out.
__device__ static void process_point_cusr(uint64_t* px, uint64_t* py, int pkField, int batch, void* vctx)
{
	VanityCtx* ctx = (VanityCtx*)vctx;
	int gpf = batch * VS_GRP_SIZE + pkField;
	uint64_t idx = ctx->starter + (uint64_t)gpf;
	uint8_t odd_py = (uint8_t)(py[0] & 1);
	bool hit = false;
	const bool endo = endomorphism_dev;
	uint64_t pe1x[4], pe2x[4], py_neg[4];
	if (endo) {
		_ModMult(pe1x, px, _beta);
		_ModMult(pe2x, px, _beta2);
		ModNeg256(py_neg, py);
	}

	if (uncompressed_dev) {
		if (endo) {
			for (int j = 0; j < 3; ++j) {
				uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
				uint8_t vbase = (uint8_t)(j * 2);
				_GetHash160Uncomp_fast(xcur, py, (uint8_t*)ctx->hash160);
				if (checkHash(ctx->hash160)) {
					vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
					vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, ctx->hash160, 20,
						vanity_make_endo_type(ENDO_GROUP_UNCOMPRESSED, (uint8_t)(vbase + ENDO_VARIANT_BASE_POS)));
				}
				_GetHash160Uncomp_fast(xcur, py_neg, (uint8_t*)ctx->hash160);
				if (checkHash(ctx->hash160)) {
					vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
					vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, ctx->hash160, 20,
						vanity_make_endo_type(ENDO_GROUP_UNCOMPRESSED, (uint8_t)(vbase + ENDO_VARIANT_BASE_NEG)));
				}
			}
		}
		else {
			_GetHash160Uncomp_fast(px, py, (uint8_t*)ctx->hash160);
			if (checkHash(ctx->hash160)) {
				vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
				vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, ctx->hash160, 20, 0x01);
			}
		}
	}

	if (compressed_dev || (segwit_dev && endo)) {
		if (endo) {
			uint32_t hash_odd[8], hash_even[8], seg_hash[8];
			for (int j = 0; j < 3; ++j) {
				uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
				uint8_t vbase = (uint8_t)(j * 2);
				uint8_t odd_variant = (uint8_t)(vbase + (odd_py ? ENDO_VARIANT_BASE_POS : ENDO_VARIANT_BASE_NEG));
				uint8_t even_variant = (uint8_t)(vbase + (odd_py ? ENDO_VARIANT_BASE_NEG : ENDO_VARIANT_BASE_POS));
				_GetHash160CompSym(xcur, (uint8_t*)hash_odd, (uint8_t*)hash_even);

				if (compressed_dev) {
					if (checkHash(hash_odd)) {
						vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
						vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, hash_odd, 20,
							vanity_make_endo_type(ENDO_GROUP_COMPRESSED, odd_variant));
					}
					if (checkHash(hash_even)) {
						vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
						vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, hash_even, 20,
							vanity_make_endo_type(ENDO_GROUP_COMPRESSED, even_variant));
					}
				}

				if (segwit_dev) {
					memcpy(seg_hash, hash_odd, 20);
					_GetHash160P2SHCompFromHash(seg_hash, seg_hash);
					if (checkHash(seg_hash)) {
						vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
						vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, seg_hash, 20,
							vanity_make_endo_type(ENDO_GROUP_SEGWIT, odd_variant));
					}
					memcpy(seg_hash, hash_even, 20);
					_GetHash160P2SHCompFromHash(seg_hash, seg_hash);
					if (checkHash(seg_hash)) {
						vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
						vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, seg_hash, 20,
							vanity_make_endo_type(ENDO_GROUP_SEGWIT, even_variant));
					}
				}
			}
		}
		else if (compressed_dev) {
			_GetHash160Comp_fast(px, odd_py, (uint8_t*)ctx->hash160);
			if (checkHash(ctx->hash160)) {
				vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
				vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, ctx->hash160, 20, 0x02);
			}
		}
	}

	if (segwit_dev && !endo) {
		if (!compressed_dev) _GetHash160Comp_fast(px, odd_py, (uint8_t*)ctx->hash160);
		_GetHash160P2SHCompFromHash(ctx->hash160, ctx->hash160);
		if (checkHash(ctx->hash160)) {
			vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
			vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, ctx->hash160, 20, 0x03);
		}
	}

	if (taproot_dev) {
		if (endo) {
			uint8_t TaprootHash[32];
			for (int j = 0; j < 3; ++j) {
				uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
				uint8_t variant = (uint8_t)(j * 2);
				u64x4_to_pubkey65(xcur, py, ctx->pubKeys);
				TweakTaproot(&TaprootHash[0], &ctx->pubKeys[0], ctx->precPtr, ctx->precPitch);
				_GetRMD160((uint32_t*)TaprootHash, ctx->hash160);
				if (checkHash(ctx->hash160)) {
					vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
					vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, TaprootHash, 32,
						vanity_make_endo_type(ENDO_GROUP_TAPROOT, variant));
				}
			}
		}
		else {
			u64x4_to_pubkey65(px, py, ctx->pubKeys);
			uint8_t TaprootHash[32];
			TweakTaproot(&TaprootHash[0], &ctx->pubKeys[0], ctx->precPtr, ctx->precPitch);
			_GetRMD160((uint32_t*)TaprootHash, ctx->hash160);
			if (checkHash(ctx->hash160)) {
				vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
				vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, TaprootHash, 32, 0x04);
			}
		}
	}

	if (xpoint_dev) {
		if (endo) {
			uint32_t xpoint_hash[8];
			for (int j = 0; j < 3; ++j) {
				uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
				uint8_t variant = (uint8_t)(j * 2);
				u64x4_to_x32(xcur, (unsigned char*)xpoint_hash);
				if (checkHash(xpoint_hash)) {
					vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
					vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, xpoint_hash, 32,
						vanity_make_endo_type(ENDO_GROUP_XPOINT, variant));
				}
			}
		}
		else {
			uint32_t xpoint_hash[8];
			u64x4_to_x32(px, (unsigned char*)xpoint_hash);
			if (checkHash(xpoint_hash)) {
				vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
				vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, xpoint_hash, 32, 0x05);
			}
		}
	}

	if (ethereum_dev) {
		if (endo) {
			uint32_t eth_hash160[5];
			for (int j = 0; j < 3; ++j) {
				uint64_t* xcur = (j == 0) ? px : ((j == 1) ? pe1x : pe2x);
				uint8_t variant = (uint8_t)(j * 2);
				keccak_eth64_tail20_from_xy(xcur, py, eth_hash160);
				if (checkHash(eth_hash160)) {
					vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
					vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, eth_hash160, 20,
						vanity_make_endo_type(ENDO_GROUP_ETH, variant));
				}
			}
		}
		else {
			uint32_t eth_hash160[5];
			keccak_eth64_tail20_from_xy(px, py, eth_hash160);
			if (checkHash(eth_hash160)) {
				vanity_cusr_cp_once(hit, ctx->c, ctx->priv_bytes, ctx->mode, idx, ctx->step);
				vanity_emit(ctx->isResult, ctx->buffResult, ctx->priv_bytes, eth_hash160, 20, 0x06);
			}
		}
	}
}

// Vanity kernel for x-only output mode.
__global__ void workerPRIV_seq_vanity_x(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, nullptr, nullptr, precPtr, precPitch };
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		vanity_walk_1024_parity(sx, sy, process_point_x, &ctx, batch);
	}
}

// Vanity kernel for compressed HASH160 mode.
__global__ void workerPRIV_seq_vanity_c(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	(void)precPtr;
	(void)precPitch;
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		if (startx_buf && starty_buf) {
			Store256A(startx_buf + blockIdx.x * 4 * blockDim.x, sx);
			Store256A(starty_buf + blockIdx.x * 4 * blockDim.x, sy);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		struct FnC {
			bool* buffResult;
			bool* isResult;
			int mode;
			uint64_t step;
			uint64_t batch_base;
			uint32_t* c;
			uint8_t* priv_bytes;
			uint32_t* hash160;
// Device helper: operator.
			__device__ __forceinline__ void operator()(uint64_t* px, uint64_t* py, int pkField) const {
				uint8_t odd_py = (uint8_t)(py[0] & 1);
				_GetHash160Comp_fast(px, odd_py, (uint8_t*)hash160);
				if (checkHash(hash160)) {
					uint64_t idx = batch_base + (uint64_t)pkField;
					vanity_save_result(c, priv_bytes, mode, idx, step, isResult, buffResult, hash160, 20, 0x02);
				}
			}
		};
		FnC fn = { buffResult, isResult, mode, step, starter + (uint64_t)(batch * VS_GRP_SIZE), c, priv_bytes, hash160 };
		vanity_walk_batch_1024_impl<false>(sx, sy, fn);
	}
	if (startx_buf && starty_buf) {
		Store256A(startx_buf + blockIdx.x * 4 * blockDim.x, sx);
		Store256A(starty_buf + blockIdx.x * 4 * blockDim.x, sy);
	}
}

// Vanity kernel for uncompressed HASH160 mode.
__global__ void workerPRIV_seq_vanity_u(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	unsigned char pubKeys[65];
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, hash160, pubKeys, precPtr, precPitch };
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		vanity_walk_1024_full_y(sx, sy, process_point_u, &ctx, batch);
	}
}

// Vanity kernel for SegWit mode.
__global__ void workerPRIV_seq_vanity_s(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	unsigned char pubKeys[65];
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, hash160, pubKeys, precPtr, precPitch };
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		vanity_walk_1024_parity(sx, sy, process_point_s, &ctx, batch);
	}
}

// Vanity kernel for Taproot mode.
__global__ void workerPRIV_seq_vanity_r(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	unsigned char pubKeys[65];
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, hash160, pubKeys, precPtr, precPitch };
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		vanity_walk_1024_full_y(sx, sy, process_point_r, &ctx, batch);
	}
}

// Vanity kernel for Ethereum mode.
__global__ void workerPRIV_seq_vanity_e(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	unsigned char pubKeys[65];
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, hash160, pubKeys, precPtr, precPitch };
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		vanity_walk_1024_full_y(sx, sy, process_point_e, &ctx, batch);
	}
}

// Vanity kernel for combined c/u/s/r/e/x mode set with optional endomorphism fan-out.
__global__ void workerPRIV_seq_vanity_cusr(
	bool* isResult, bool* buffResult,
	const secp256k1_ge_storage* __restrict__ precPtr, size_t precPitch,
	int mode, uint8_t* start_point, uint64_t step, int thread_steps_pub,
	uint64_t* __restrict__ startx_buf, uint64_t* __restrict__ starty_buf)
{
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t starter = thread_steps_pub * (uint64_t)tIx;
	uint32_t c[8];
	loadBE256toWords(start_point, c);
	uint8_t priv_bytes[32];
	uint64_t sx[4], sy[4];
	if (startx_buf && starty_buf) {
		Load256A(sx, startx_buf + blockIdx.x * 4 * blockDim.x);
		Load256A(sy, starty_buf + blockIdx.x * 4 * blockDim.x);
	} else {
		secp256k1_gej Pstart;
		secp256k1_walk_get_start_gej(&Pstart, starter + (uint64_t)VS_GRP_HALF);
		secp256k1_ge ge_start;
		secp256k1_ge_set_gej(&ge_start, &Pstart);
		fe_to_u64x4(sx, &ge_start.x);
		fe_to_u64x4(sy, &ge_start.y);
	}
	uint32_t hash160[8];
	unsigned char pubKeys[65];
	VanityCtx ctx = { c, priv_bytes, mode, step, starter, isResult, buffResult, hash160, pubKeys, precPtr, precPitch };
	bool need_full_y = uncompressed_dev || taproot_dev || ethereum_dev;
	int numBatches = thread_steps_pub / VS_GRP_SIZE;
	if (!vanity_filters_active()) {
		VanityNoopPoint fn_noop;
		for (int batch = 0; batch < numBatches; batch++) {
			vanity_walk_batch_1024_impl<false>(sx, sy, fn_noop);
		}
		return;
	}
	for (int batch = 0; batch < numBatches; batch++) {
		if (need_full_y)
			vanity_walk_1024_full_y(sx, sy, process_point_cusr, &ctx, batch);
		else
			vanity_walk_1024_parity(sx, sy, process_point_cusr, &ctx, batch);
	}
}
