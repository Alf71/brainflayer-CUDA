#include "big_int_device.cuh"

//----------------------------------------
// Inline PTX macros for add/sub
//----------------------------------------
#define madc_hi(dest, a, x, b)     asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t"     : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_hi_cc(dest, a, x, b)  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t"  : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define mad_hi_cc(dest, a, x, b)   asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t"   : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define mad_lo_cc(dest, a, x, b)   asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t"   : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo(dest, a, x, b)     asm volatile("madc.lo.u32 %0, %1, %2, %3;\n\t"     : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo_cc(dest, a, x, b)  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t"  : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define addc(dest, a, b)           asm volatile("addc.u32 %0, %1, %2;\n\t"            : "=r"(dest) : "r"(a), "r"(b))
#define add_cc(dest, a, b)         asm volatile("add.cc.u32 %0, %1, %2;\n\t"          : "=r"(dest) : "r"(a), "r"(b))
#define addc_cc(dest, a, b)        asm volatile("addc.cc.u32 %0, %1, %2;\n\t"         : "=r"(dest) : "r"(a), "r"(b))

#define sub_cc(dest, a, b)         asm volatile("sub.cc.u32 %0, %1, %2;\n\t"          : "=r"(dest) : "r"(a), "r"(b))
#define subc_cc(dest, a, b)        asm volatile("subc.cc.u32 %0, %1, %2;\n\t"         : "=r"(dest) : "r"(a), "r"(b))
#define subc(dest, a, b)           asm volatile("subc.u32 %0, %1, %2;\n\t"            : "=r"(dest) : "r"(a), "r"(b))

#define set_eq(dest,a,b)           asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t"      : "=r"(dest) : "r"(a), "r"(b))

#define lsbpos(x) (__ffs((x)))



//----------------------------------------
// Helpers for big-endian <-> 32-bit words
//----------------------------------------

// 256-bit => c[8]
__device__  void loadBE256toWords(const uint8_t src[32], uint32_t c[8])
{
#pragma unroll 8
    for (int i = 0; i < 8; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE256.
__device__  void storeWordsToBE256(const uint32_t c[8], uint8_t dst[32])
{
#pragma unroll 8
    for (int i = 0; i < 8; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

// 512-bit => c[16]
__device__  void loadBE512toWords(const uint8_t src[64], uint32_t c[16])
{
#pragma unroll 16
    for (int i = 0; i < 16; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE512.
__device__  void storeWordsToBE512(const uint32_t c[16], uint8_t dst[64])
{
#pragma unroll 16
    for (int i = 0; i < 16; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

// 1024-bit => c[32]
__device__  void loadBE1024toWords(const uint8_t src[128], uint32_t c[32])
{
#pragma unroll 32
    for (int i = 0; i < 32; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE1024.
__device__  void storeWordsToBE1024(const uint32_t c[32], uint8_t dst[128])
{
#pragma unroll 32
    for (int i = 0; i < 32; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

// 2048-bit => c[64]
__device__  void loadBE2048toWords(const uint8_t src[256], uint32_t c[64])
{
#pragma unroll 64
    for (int i = 0; i < 64; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE2048.
__device__  void storeWordsToBE2048(const uint32_t c[64], uint8_t dst[256])
{
#pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

// 4096-bit => c[128]
__device__  void loadBE4096toWords(const uint8_t src[512], uint32_t c[128])
{
#pragma unroll 128
    for (int i = 0; i < 128; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE4096.
__device__  void storeWordsToBE4096(const uint32_t c[128], uint8_t dst[512])
{
#pragma unroll 128
    for (int i = 0; i < 128; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

// 8192-bit => c[256]
__device__  void loadBE8192toWords(const uint8_t src[1024], uint32_t c[256])
{
#pragma unroll 256
    for (int i = 0; i < 256; i++) {
        c[i] = ((uint32_t)src[4 * i + 0] << 24)
            | ((uint32_t)src[4 * i + 1] << 16)
            | ((uint32_t)src[4 * i + 2] << 8)
            | ((uint32_t)src[4 * i + 3] << 0);
    }
}
// Device helper: storeWordsToBE8192.
__device__  void storeWordsToBE8192(const uint32_t c[256], uint8_t dst[1024])
{
#pragma unroll 256
    for (int i = 0; i < 256; i++) {
        uint32_t w = c[i];
        dst[4 * i + 0] = (uint8_t)(w >> 24);
        dst[4 * i + 1] = (uint8_t)(w >> 16);
        dst[4 * i + 2] = (uint8_t)(w >> 8);
        dst[4 * i + 3] = (uint8_t)(w >> 0);
    }
}

//----------------------------------------
//----------------------------------------
__device__  int add64to256_device(const uint32_t in[8],
    uint64_t val64,
    uint32_t out[8])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    // LSW = out[7], MSW = out[0]
    add_cc(out[7], in[7], lo);
    addc_cc(out[6], in[6], hi);
    addc_cc(out[5], in[5], 0);
    addc_cc(out[4], in[4], 0);
    addc_cc(out[3], in[3], 0);
    addc_cc(out[2], in[2], 0);
    addc_cc(out[1], in[1], 0);
    addc_cc(out[0], in[0], 0);

    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry; // 1 => overflow
}

// Device helper: mul_u64_u64_to_u128.
__device__ void mul_u64_u64_to_u128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    lo = a * b;
    hi = __umul64hi(a, b);
}

// Device helper: add128to256_device.
__device__ void add128to256_device(const uint32_t c[8], uint64_t lo, uint64_t hi, uint32_t r[8]) {
    uint32_t v0 = (uint32_t)(lo);
    uint32_t v1 = (uint32_t)(lo >> 32);
    uint32_t v2 = (uint32_t)(hi);
    uint32_t v3 = (uint32_t)(hi >> 32);

    uint64_t sum;
    uint64_t carry = 0;

    // add to least significant 128 bits => words 7..4
    sum = (uint64_t)c[7] + v0;            r[7] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[6] + v1 + carry;    r[6] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[5] + v2 + carry;    r[5] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[4] + v3 + carry;    r[4] = (uint32_t)sum; carry = sum >> 32;

    // propagate carry into higher words
    sum = (uint64_t)c[3] + carry;         r[3] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[2] + carry;         r[2] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[1] + carry;         r[1] = (uint32_t)sum; carry = sum >> 32;
    sum = (uint64_t)c[0] + carry;         r[0] = (uint32_t)sum;
}

__device__  int sub64from256_device(const uint32_t in[8],
    uint64_t val64,
    uint32_t out[8])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[7], in[7], lo);
    subc_cc(out[6], in[6], hi);
    subc_cc(out[5], in[5], 0);
    subc_cc(out[4], in[4], 0);
    subc_cc(out[3], in[3], 0);
    subc_cc(out[2], in[2], 0);
    subc_cc(out[1], in[1], 0);
    subc_cc(out[0], in[0], 0);

    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1); // 1 => < 0 mod 2^256
}


// Device helper: sub128from256_device.
__device__ void sub128from256_device(const uint32_t c[8], uint64_t lo, uint64_t hi, uint32_t r[8]) {
    uint32_t v0 = (uint32_t)(lo);
    uint32_t v1 = (uint32_t)(lo >> 32);
    uint32_t v2 = (uint32_t)(hi);
    uint32_t v3 = (uint32_t)(hi >> 32);

    uint64_t borrow = 0;
    uint64_t diff;

    auto sub32 = [&](uint32_t a, uint32_t b) -> uint32_t {
        uint64_t x = (uint64_t)a;
        uint64_t y = (uint64_t)b + borrow;
        if (x >= y) { diff = x - y; borrow = 0; }
        else { diff = (1ULL << 32) + x - y; borrow = 1; }
        return (uint32_t)diff;
        };

    // subtract from least significant 128 bits => words 7..4
    r[7] = sub32(c[7], v0);
    r[6] = sub32(c[6], v1);
    r[5] = sub32(c[5], v2);
    r[4] = sub32(c[4], v3);

    // propagate borrow
    r[3] = sub32(c[3], 0);
    r[2] = sub32(c[2], 0);
    r[1] = sub32(c[1], 0);
    r[0] = sub32(c[0], 0);
}


//----------------------------------------
//----------------------------------------
__device__  int add64to512_device(const uint32_t in[16],
    uint64_t val64,
    uint32_t out[16])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    add_cc(out[15], in[15], lo);
    addc_cc(out[14], in[14], hi);
#pragma unroll
    for (int i = 13; i >= 0; i--) {
        addc_cc(out[i], in[i], 0);
    }

    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry;
}

__device__  int sub64from512_device(const uint32_t in[16],
    uint64_t val64,
    uint32_t out[16])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[15], in[15], lo);
    subc_cc(out[14], in[14], hi);
#pragma unroll
    for (int i = 13; i >= 0; i--) {
        subc_cc(out[i], in[i], 0);
    }

    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1);
}

//----------------------------------------
//----------------------------------------
__device__  int add64to1024_device(const uint32_t in[32],
    uint64_t val64,
    uint32_t out[32])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    add_cc(out[31], in[31], lo);
    addc_cc(out[30], in[30], hi);
#pragma unroll
    for (int i = 29; i >= 0; i--) {
        addc_cc(out[i], in[i], 0);
    }
    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry;
}

__device__  int sub64from1024_device(const uint32_t in[32],
    uint64_t val64,
    uint32_t out[32])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[31], in[31], lo);
    subc_cc(out[30], in[30], hi);
#pragma unroll
    for (int i = 29; i >= 0; i--) {
        subc_cc(out[i], in[i], 0);
    }
    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1);
}

//----------------------------------------
//----------------------------------------
__device__  int add64to2048_device(const uint32_t in[64],
    uint64_t val64,
    uint32_t out[64])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    add_cc(out[63], in[63], lo);
    addc_cc(out[62], in[62], hi);
#pragma unroll
    for (int i = 61; i >= 0; i--) {
        addc_cc(out[i], in[i], 0);
    }
    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry;
}

__device__  int sub64from2048_device(const uint32_t in[64],
    uint64_t val64,
    uint32_t out[64])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[63], in[63], lo);
    subc_cc(out[62], in[62], hi);
#pragma unroll
    for (int i = 61; i >= 0; i--) {
        subc_cc(out[i], in[i], 0);
    }
    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1);
}

//----------------------------------------
//----------------------------------------
__device__  int add64to4096_device(const uint32_t in[128],
    uint64_t val64,
    uint32_t out[128])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    add_cc(out[127], in[127], lo);
    addc_cc(out[126], in[126], hi);
#pragma unroll
    for (int i = 125; i >= 0; i--) {
        addc_cc(out[i], in[i], 0);
    }
    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry;
}

__device__  int sub64from4096_device(const uint32_t in[128],
    uint64_t val64,
    uint32_t out[128])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[127], in[127], lo);
    subc_cc(out[126], in[126], hi);
#pragma unroll
    for (int i = 125; i >= 0; i--) {
        subc_cc(out[i], in[i], 0);
    }
    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1);
}

//----------------------------------------
//----------------------------------------
__device__  int add64to8192_device(const uint32_t in[256],
    uint64_t val64,
    uint32_t out[256])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    add_cc(out[255], in[255], lo);
    addc_cc(out[254], in[254], hi);
#pragma unroll
    for (int i = 253; i >= 0; i--) {
        addc_cc(out[i], in[i], 0);
    }
    unsigned int carry = 0;
    addc(carry, 0, 0);
    return carry;
}

__device__  int sub64from8192_device(const uint32_t in[256],
    uint64_t val64,
    uint32_t out[256])
{
    uint32_t hi = (uint32_t)(val64 >> 32);
    uint32_t lo = (uint32_t)(val64 & 0xFFFFFFFF);

    sub_cc(out[255], in[255], lo);
    subc_cc(out[254], in[254], hi);
#pragma unroll
    for (int i = 253; i >= 0; i--) {
        subc_cc(out[i], in[i], 0);
    }
    unsigned int borrow = 0;
    subc(borrow, 0, 0);
    return (borrow & 1);
}

//----------------------------------------
//----------------------------------------
__device__  int sizeBE256_device(const uint32_t c[8])
{
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int leadingZeroBytes = 0;
            if ((w >> 24) == 0) {
                leadingZeroBytes++;
                w <<= 8;
                if ((w >> 24) == 0) {
                    leadingZeroBytes++;
                    w <<= 8;
                    if ((w >> 24) == 0) {
                        leadingZeroBytes++;
                        w <<= 8;
                        if ((w >> 24) == 0) {
                            leadingZeroBytes++;
                        }
                    }
                }
            }
            int totalBytes = (8 - i) * 4 - leadingZeroBytes;
            return totalBytes;
        }
    }
    return 0;
}

// Device helper: sizeBE512_device.
__device__  int sizeBE512_device(const uint32_t c[16])
{
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int leadingZeroBytes = 0;
            if ((w >> 24) == 0) {
                leadingZeroBytes++;
                w <<= 8;
                if ((w >> 24) == 0) {
                    leadingZeroBytes++;
                    w <<= 8;
                    if ((w >> 24) == 0) {
                        leadingZeroBytes++;
                        w <<= 8;
                        if ((w >> 24) == 0) {
                            leadingZeroBytes++;
                        }
                    }
                }
            }
            int totalBytes = (16 - i) * 4 - leadingZeroBytes;
            return totalBytes;
        }
    }
    return 0;
}

// 1024-bit
__device__  int sizeBE1024_device(const uint32_t c[32])
{
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int lzb = 0;
            if ((w >> 24) == 0) {
                lzb++; w <<= 8;
                if ((w >> 24) == 0) {
                    lzb++; w <<= 8;
                    if ((w >> 24) == 0) {
                        lzb++; w <<= 8;
                        if ((w >> 24) == 0) { lzb++; }
                    }
                }
            }
            int totalBytes = (32 - i) * 4 - lzb;
            return totalBytes;
        }
    }
    return 0;
}

// 2048-bit
__device__  int sizeBE2048_device(const uint32_t c[64])
{
#pragma unroll
    for (int i = 0; i < 64; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int lzb = 0;
            if ((w >> 24) == 0) {
                lzb++; w <<= 8;
                if ((w >> 24) == 0) {
                    lzb++; w <<= 8;
                    if ((w >> 24) == 0) {
                        lzb++; w <<= 8;
                        if ((w >> 24) == 0) { lzb++; }
                    }
                }
            }
            int totalBytes = (64 - i) * 4 - lzb;
            return totalBytes;
        }
    }
    return 0;
}

// 4096-bit
__device__  int sizeBE4096_device(const uint32_t c[128])
{
#pragma unroll
    for (int i = 0; i < 128; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int lzb = 0;
            if ((w >> 24) == 0) {
                lzb++; w <<= 8;
                if ((w >> 24) == 0) {
                    lzb++; w <<= 8;
                    if ((w >> 24) == 0) {
                        lzb++; w <<= 8;
                        if ((w >> 24) == 0) { lzb++; }
                    }
                }
            }
            int totalBytes = (128 - i) * 4 - lzb;
            return totalBytes;
        }
    }
    return 0;
}

// 8192-bit
__device__  int sizeBE8192_device(const uint32_t c[256])
{
#pragma unroll
    for (int i = 0; i < 256; i++)
    {
        uint32_t w = c[i];
        if (w != 0)
        {
            int lzb = 0;
            if ((w >> 24) == 0) {
                lzb++; w <<= 8;
                if ((w >> 24) == 0) {
                    lzb++; w <<= 8;
                    if ((w >> 24) == 0) {
                        lzb++; w <<= 8;
                        if ((w >> 24) == 0) { lzb++; }
                    }
                }
            }
            int totalBytes = (256 - i) * 4 - lzb;
            return totalBytes;
        }
    }
    return 0;
}

//----------------------------------------
//----------------------------------------
__device__  int cmp256_device(const uint32_t A[8],
    const uint32_t B[8])
{
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}

__device__  int cmp512_device(const uint32_t A[16],
    const uint32_t B[16])
{
#pragma unroll
    for (int i = 0; i < 16; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}
__device__  int cmp1024_device(const uint32_t A[32],
    const uint32_t B[32])
{
#pragma unroll
    for (int i = 0; i < 32; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}
__device__  int cmp2048_device(const uint32_t A[64],
    const uint32_t B[64])
{
#pragma unroll
    for (int i = 0; i < 64; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}
__device__  int cmp4096_device(const uint32_t A[128],
    const uint32_t B[128])
{
#pragma unroll
    for (int i = 0; i < 128; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}
__device__  int cmp8192_device(const uint32_t A[256],
    const uint32_t B[256])
{
#pragma unroll
    for (int i = 0; i < 256; i++) {
        if (A[i] < B[i]) return -1;
        if (A[i] > B[i]) return 1;
    }
    return 0;
}

// Device helper: bump_key_256.
__device__ bool bump_key_256(uint8_t* __restrict__ prv32, uint64_t n, bool plus)
{
    bool is_null = false;
    __align__(16) uint32_t c[8], r[8];
    loadBE256toWords(prv32, c);
    if (plus) add64to256_device(c, n, r);
    else      sub64from256_device(c, n, r);

    uint32_t orv = r[0] | r[1] | r[2] | r[3] | r[4] | r[5] | r[6] | r[7];
    if (plus && orv == 0) {
        r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0; r[4] = 0; r[5] = 0; r[6] = 0; r[7] = 2;

        is_null = true;
    }

    storeWordsToBE256(r, prv32);

    return is_null;
}



// Device helper: bump_all_keys.
__device__ int bump_all_keys(uint8_t* __restrict__ prvKeys, int count, uint64_t n, bool plus, int* zeroz)
{

    __align__(16) uint32_t c[8];
    __align__(16) uint32_t r[8];
    int wrapped = 0;

    for (int i = 0; i < count; i++) {

        
        uint8_t* pvk = &prvKeys[i * 32];

        loadBE256toWords(pvk, c);
        if (plus) add64to256_device(c, n, r);
        else      sub64from256_device(c, n, r);



        uint32_t orv = r[0] | r[1] | r[2] | r[3] | r[4] | r[5] | r[6] | r[7];
        if (plus && orv == 0) {
            r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0; r[4] = 0; r[5] = 0; r[6] = 0; r[7] = 2;

            if(zeroz)
            { 
                zeroz[i] = i + 1;
            }
            wrapped++;
        }
        storeWordsToBE256(r, pvk);

    }
    return wrapped;
}
