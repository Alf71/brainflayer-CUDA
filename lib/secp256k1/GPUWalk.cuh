#pragma once
// VanitySearch-style ECC walk - Block B from OPTIMIZATION_PLAN.txt
// Uses 4-limb uint64[4] arithmetic with Montgomery batch inversion (1 inv per 1024 points)

#include "secp256k1_common.cuh"
#include "secp256k1_field.cuh"
#include "secp256k1_group.cuh"
#include "GPUMath.cuh"
#include "GPUGroup.cuh"
#include <cstdint>

#define VS_GRP_SIZE 1024
#define VS_GRP_HALF (VS_GRP_SIZE / 2)

// Convert secp256k1_fe (10x26-bit limbs) -> uint64[4] (VanitySearch format)
// VanitySearch: r[0]=LSB, r[3]=MSB; big-endian bytes within each 64-bit word
static __device__ void fe_to_u64x4(uint64_t r[4], const secp256k1_fe* fe) {
    __align__(16) unsigned char buf[32];
    __align__(16) secp256k1_fe tmp = *fe;
    secp256k1_fe_normalize_var(&tmp);
    secp256k1_fe_get_b32(buf, &tmp);
    for (int w = 0; w < 4; w++) {
        uint64_t val = 0;
        for (int b = 0; b < 8; b++)
            val = (val << 8) | buf[(3 - w) * 8 + b];
        r[w] = val;
    }
}

// Convert uint64[4] -> secp256k1_fe (for P0_global update)
static __device__ void u64x4_to_fe(secp256k1_fe* r, const uint64_t x[4]) {
    __align__(16) unsigned char buf[32];
    for (int w = 0; w < 4; w++) {
        uint64_t val = x[w];
        for (int b = 7; b >= 0; b--) {
            buf[(3 - w) * 8 + b] = (unsigned char)(val & 0xFF);
            val >>= 8;
        }
    }
    secp256k1_fe_set_b32(r, buf);
}

// VanitySearch-style ECC walk: process 1024 points with single batch inversion.
// sx,sy: start point (affine, uint64[4]). On return: sx,sy = start + 1024*G.
// fn_process_point(px, py, pkField): px=X, py=full Y or py[0]=odd_py when !NeedFullY.
// NeedFullY=false: use ModSub256isOdd (parity only) for c/x - avoids full Y reduction.
template<bool NeedFullY, typename Fn>
__device__ void vanity_walk_batch_1024_impl(
    uint64_t sx[4], uint64_t sy[4],
    Fn&& fn_process_point)
{
    __align__(16) uint64_t dx[4], px[4], py[4], py_full[4], dy[4];
    __align__(16) uint64_t sxn[4], syn[4], sx_gx[4];
    uint8_t odd_py;
    __align__(16) uint64_t inverse[5];
    __align__(16) uint64_t subp[VS_GRP_HALF][4];

    // Check starting point (center of group = P)
    Load256(py_full, sy);
    fn_process_point(sx, py_full, VS_GRP_HALF);

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

        // Positive point: P + (i+1)*G
        ModSub256(dy, Gy[i], sy);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);
        ModSub256(py, sx, px);
        _ModMult(py, dy);
        if (NeedFullY) {
            ModSub256(py_full, py, sy);
        } else {
            ModSub256isOdd(py, sy, &odd_py);
            py_full[0] = (py_full[0] & ~1ULL) | odd_py;
        }
        fn_process_point(px, py_full, VS_GRP_HALF + (i + 1));

        // Negative point: P - (i+1)*G
        ModSub256(dy, syn, Gy[i]);
        _ModMult(dy, dx);
        _ModSqr(px, dy);
        ModSub256(px, sx_gx);
        ModSub256(py, px, sx);
        _ModMult(py, dy);
        if (NeedFullY) {
            ModNeg256(py_full, py);
            ModSub256(py_full, py_full, sy);
        } else {
            ModSub256isOdd(syn, py, &odd_py);
            py_full[0] = (py_full[0] & ~1ULL) | odd_py;
        }
        fn_process_point(px, py_full, VS_GRP_HALF - (i + 1));

        ModSub256(dx, Gx[i], sx);
        _ModMult(inverse, dx);
    }

    // Last negative point (P - 512*G, pkField=0)
    _ModMult(dx, subp[i], inverse);
    ModSub256(dy, syn, Gy[i]);
    _ModMult(dy, dx);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, Gx[i]);
    ModSub256(py, px, sx);
    _ModMult(py, dy);
    if (NeedFullY) {
        ModNeg256(py_full, py);
        ModSub256(py_full, py_full, sy);
    } else {
        ModSub256isOdd(syn, py, &odd_py);
        py_full[0] = (py_full[0] & ~1ULL) | odd_py;
    }
    fn_process_point(px, py_full, 0);

    // Compute new start point: P + 1024*G for next batch
    ModSub256(dy, _2Gny, sy);
    ModSub256(dx, Gx[i], sx);
    _ModMult(inverse, dx);
    _ModMult(dy, inverse);
    _ModSqr(px, dy);
    ModSub256(px, sx);
    ModSub256(px, _2Gnx);
    ModSub256(py, _2Gnx, px);  // py = _2Gnx - px
    _ModMult(py, dy);          // py = dy * (_2Gnx - px)
    ModSub256(py, _2Gny);      // py = dy*(_2Gnx-px) - _2Gny = y3

    Load256(sx, px);
    Load256(sy, py);
}

template<typename Fn>
// Device helper: vanity_walk_batch_1024.
__device__ void vanity_walk_batch_1024(uint64_t sx[4], uint64_t sy[4], Fn&& fn) {
    vanity_walk_batch_1024_impl<true>(sx, sy, fn);
}

// Non-template versions - implemented in WorkerPRIV_vanity.cu
typedef void (*vanity_process_point_fn)(uint64_t* px, uint64_t* py, int pkField, int batch, void* ctx);
__device__ void vanity_walk_1024_parity(uint64_t sx[4], uint64_t sy[4], vanity_process_point_fn fn, void* ctx, int batch);
__device__ void vanity_walk_1024_full_y(uint64_t sx[4], uint64_t sy[4], vanity_process_point_fn fn, void* ctx, int batch);
