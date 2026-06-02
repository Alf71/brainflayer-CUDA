#pragma once

#ifndef UINT128_T_H
#define UINT128_T_H

#include <cstdint>
#include <iosfwd>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cstdio>
#include <cstdlib>
#endif

#if defined(__GNUC__) || defined(__clang__)

__device__ __host__ static inline uint64_t _lzcnt_u64(uint64_t x) {
    if (x == 0) return 64;
    return __builtin_clzll(x);
}

// _tzcnt_u64(x) -> __builtin_ctzll(x)
__device__ __host__ static inline uint64_t _tzcnt_u64(uint64_t x) {
    if (x == 0) return 64;
    return __builtin_ctzll(x);
}

__device__ __host__ static inline uint64_t _bzhi_u64(uint64_t x, uint32_t n) {
    if (n >= 64) return x;
    if (n == 0) return 0ULL;
    return x & ((1ULL << n) - 1ULL);
}

static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t* high) {
    __uint128_t res = (__uint128_t)a * (__uint128_t)b;
    *high = (uint64_t)(res >> 64);
    return (uint64_t)res;
}

__device__ __host__ __inline__ static unsigned char addcarry_u64(
    unsigned char Carry,
    unsigned long Source1,
    unsigned long Source2,
    unsigned long* Destination
)
{
    unsigned long Sum = (Carry != 0) + Source1 + Source2;
    unsigned long CarryVector = (Source1 & Source2) ^ ((Source1 ^ Source2) & ~Sum);
    *Destination = Sum;
    return CarryVector >> 63;
}

__device__ __host__ static unsigned char _subborrow_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* res) {
    uint64_t sub = a - b - c;
    *res = sub;
    return (sub > a) ? 1 : 0;
}


static inline int __clzll(uint64_t x) { return (int)_lzcnt_u64(x); }
static inline int __ctzll(uint64_t x) { return (int)_tzcnt_u64(x); }

#else
__device__ __host__ __inline__ static unsigned char addcarry_u64(
    unsigned char Carry,
    unsigned __int64 Source1,
    unsigned __int64 Source2,
    unsigned __int64* Destination
)
{
    unsigned __int64 Sum = (Carry != 0) + Source1 + Source2;
    unsigned __int64 CarryVector = (Source1 & Source2) ^ ((Source1 ^ Source2) & ~Sum);
    *Destination = Sum;
    return CarryVector >> 63;
}

__device__ __host__ static unsigned char _subborrow_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* res) {
    uint64_t sub = a - b - c;
    *res = sub;
    return (sub > a) ? 1 : 0;
}
// lzcnt64 -> __lzcnt64
// tzcnt64 -> __tzcnt_u64
// bzhi_u64 -> __bzhi_u64



#endif


#define MAKE_BINARY_OP_HELPERS(op) \
friend __device__ __host__ auto operator op(const uint128_t& x, uint8_t   y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, uint16_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, uint32_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, uint64_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, int8_t   y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, int16_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, int32_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, int64_t  y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(const uint128_t& x, char y) { return operator op(x, (uint128_t)y); }  \
friend __device__ __host__ auto operator op(uint8_t   x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(uint16_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(uint32_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(uint64_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(int8_t   x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(int16_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(int32_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(int64_t  x, const uint128_t& y) { return operator op((uint128_t)x, y); }  \
friend __device__ __host__ auto operator op(char x, const uint128_t& y) { return operator op((uint128_t)x, y); }

#define MAKE_BINARY_OP_HELPERS_FLOAT(op) \
friend __device__ __host__ auto operator op(const uint128_t& x, float  y) { return (float)x op y; }   \
friend __device__ __host__ auto operator op(const uint128_t& x, double y) { return (double)x op y; }  \
friend __device__ __host__ auto operator op(float  x, const uint128_t& y) { return x op (float)y; }    \
friend __device__ __host__ auto operator op(double x, const uint128_t& y) { return x op (double)y; }

#define MAKE_BINARY_OP_HELPERS_uint64_t(op) \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, uint8_t  n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, uint16_t n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, uint32_t n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, int8_t  n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, int16_t n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, int32_t n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, int64_t n) { return operator op(x, (uint64_t)n); }    \
friend __device__ __host__ uint128_t operator op(const uint128_t& x, const uint128_t& n) { return operator op(x, (uint64_t)n); }

class uint128_t
{
public:

    uint64_t m_lo;
    uint64_t m_hi;
    friend __device__ __host__ uint128_t DivMod(uint128_t n, uint128_t d, uint128_t& rem);

    __device__ __host__ uint128_t() {}
    __device__ __host__ uint128_t(uint8_t    x) : m_lo(x), m_hi(0) {}
    __device__ __host__ uint128_t(uint16_t   x) : m_lo(x), m_hi(0) {}
    __device__ __host__ uint128_t(uint32_t   x) : m_lo(x), m_hi(0) {}
    __device__ __host__ uint128_t(uint64_t   x) : m_lo(x), m_hi(0) {}
    __device__ __host__ uint128_t(int8_t    x) : m_lo((uint64_t)(int64_t)x), m_hi((uint64_t)((int64_t)x >> 63)) {}
    __device__ __host__ uint128_t(int16_t   x) : m_lo((uint64_t)(int64_t)x), m_hi((uint64_t)((int64_t)x >> 63)) {}
    __device__ __host__ uint128_t(int32_t   x) : m_lo((uint64_t)(int64_t)x), m_hi((uint64_t)((int64_t)x >> 63)) {}
    __device__ __host__ uint128_t(int64_t   x) : m_lo((uint64_t)x), m_hi((uint64_t)(x >> 63)) {}
    __device__ __host__ uint128_t(uint64_t hi, uint64_t lo) : m_lo(lo), m_hi(hi) {}

    __device__ __host__ static uint64_t __shiftleft128(uint64_t lo, uint64_t hi, uint8_t n) {
        if (n >= 64) {
            return hi << (n - 64);
        }
        else {
            return (hi << n) | (lo >> (64 - n));
        }
    }

    __device__ __host__ static uint64_t __shiftright128(uint64_t lo, uint64_t hi, uint8_t n) {
        if (n >= 64) {
            return lo >> (n - 64);
        }
        else {
            return (hi << (64 - n)) | (lo >> n);
        }
    }

#ifdef _MSC_VER
    __device__ __host__ static unsigned char _subborrow_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* res) {
        uint64_t sub = a - b - c;
        *res = sub;
        return (sub > a) ? 1 : 0;
    }
#else
    // __device__ __host__ static unsigned char _subborrow_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* res);
#endif

    __device__ __host__ static uint64_t _udiv64(uint64_t n, uint64_t d) {
        return n / d;
    }

    __device__ __host__ static uint64_t _udiv128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t* rem) {
        uint128_t num = { hi, lo };
        uint128_t den = { 0, d };
        uint128_t quotient = num / den;
        *rem = (num % den).m_lo;
        return quotient.m_lo;
    }


    __host__ uint128_t(float x);
    __host__ uint128_t(double x);

#ifdef _MSC_VER
#else
#endif

    __device__ __host__ uint128_t& operator+=(const uint128_t& x)
    {
        addcarry_u64(addcarry_u64(0, m_lo, x.m_lo, &m_lo), m_hi, x.m_hi, &m_hi);
        return *this;
    }

    friend __device__ __host__ uint128_t operator+(const uint128_t& x, const uint128_t& y)
    {
        uint128_t ret;
        addcarry_u64(addcarry_u64(0, x.m_lo, y.m_lo, &ret.m_lo), x.m_hi, y.m_hi, &ret.m_hi);
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(+);
    MAKE_BINARY_OP_HELPERS_FLOAT(+);

    __device__ __host__ uint128_t& operator-=(const uint128_t& x)
    {
        _subborrow_u64(_subborrow_u64(0, m_lo, x.m_lo, &m_lo), m_hi, x.m_hi, &m_hi);
        return *this;
    }

    friend __device__ __host__ uint128_t operator-(const uint128_t& x, const uint128_t& y)
    {
        uint128_t ret;
        _subborrow_u64(_subborrow_u64(0, x.m_lo, y.m_lo, &ret.m_lo), x.m_hi, y.m_hi, &ret.m_hi);
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(-);
    MAKE_BINARY_OP_HELPERS_FLOAT(-);

    __device__ __host__ static inline uint128_t mul128(uint64_t x, uint64_t y)
    {
        uint128_t res;
#ifdef __CUDA_ARCH__
        res.m_lo = x * y;
        res.m_hi = __umul64hi(x, y);
#else
        res.m_lo = _umul128(x, y, &res.m_hi);
#endif
        return res;
    }

    __device__ __host__ uint128_t& operator*=(const uint128_t& x)
    {
#ifdef __CUDA_ARCH__
        uint128_t temp = mul128(m_lo, x.m_lo);
        m_hi = temp.m_hi + m_hi * x.m_lo + m_lo * x.m_hi;
        m_lo = temp.m_lo;
        return *this;
#else
        uint64_t hHi;
        const uint64_t hLo = _umul128(m_lo, x.m_lo, &hHi);
        m_hi = hHi + m_hi * x.m_lo + m_lo * x.m_hi;
        m_lo = hLo;
        return *this;
#endif
    }

    friend __device__ __host__ uint128_t operator*(const uint128_t& x, const uint128_t& y)
    {
#ifdef __CUDA_ARCH__
        uint128_t ret = mul128(x.m_lo, y.m_lo);
        ret.m_hi += y.m_hi * x.m_lo + y.m_lo * x.m_hi;
        return ret;
#else
        uint128_t ret;
        uint64_t hHi;
        ret.m_lo = _umul128(x.m_lo, y.m_lo, &hHi);
        ret.m_hi = hHi + y.m_hi * x.m_lo + y.m_lo * x.m_hi;
        return ret;
#endif
    }

    MAKE_BINARY_OP_HELPERS(*);
    MAKE_BINARY_OP_HELPERS_FLOAT(*);

    __device__ __host__ uint128_t& operator/=(const uint128_t& x)
    {
        uint128_t rem;
        *this = DivMod(*this, x, rem);
        return *this;
    }

    friend __device__ __host__ uint128_t operator/(const uint128_t& x, const uint128_t& y)
    {
        uint128_t rem;
        return DivMod(x, y, rem);
    }

    MAKE_BINARY_OP_HELPERS(/ );
    MAKE_BINARY_OP_HELPERS_FLOAT(/ );

    __device__ __host__ uint128_t& operator%=(const uint128_t& x)
    {
        DivMod(*this, x, *this);
        return *this;
    }

    friend __device__ __host__ uint128_t operator%(const uint128_t& x, const uint128_t& y)
    {
        uint128_t ret;
        DivMod(x, y, ret);
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(%);

    __device__ __host__ uint128_t& operator&=(const uint128_t& x)
    {
        m_hi &= x.m_hi;
        m_lo &= x.m_lo;
        return *this;
    }

    friend __device__ __host__ uint128_t operator&(const uint128_t& x, const uint128_t& y)
    {
        return uint128_t(x.m_hi & y.m_hi, x.m_lo & y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(&);

    __device__ __host__ uint128_t& operator|=(const uint128_t& x)
    {
        m_hi |= x.m_hi;
        m_lo |= x.m_lo;
        return *this;
    }

    friend __device__ __host__ uint128_t operator|(const uint128_t& x, const uint128_t& y)
    {
        return uint128_t(x.m_hi | y.m_hi, x.m_lo | y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(| );

    __device__ __host__ uint128_t& operator^=(const uint128_t& x)
    {
        m_hi ^= x.m_hi;
        m_lo ^= x.m_lo;
        return *this;
    }

    friend __device__ __host__ uint128_t operator^(const uint128_t& x, const uint128_t& y)
    {
        return uint128_t(x.m_hi ^ y.m_hi, x.m_lo ^ y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(^);

    __device__ __host__ uint128_t& operator>>=(uint64_t n)
    {
        const uint64_t lo = __shiftright128(m_lo, m_hi, (uint8_t)n);
        const uint64_t hi = m_hi >> (n & 63ULL);

        m_lo = (n & 64) ? hi : lo;
        m_hi = (n & 64) ? 0 : hi;

        return *this;
    }

    friend __device__ __host__ uint128_t operator>>(const uint128_t& x, uint64_t n)
    {
        uint128_t ret;

        const uint64_t lo = __shiftright128(x.m_lo, x.m_hi, (uint8_t)n);
        const uint64_t hi = x.m_hi >> (n & 63ULL);

        ret.m_lo = (n & 64) ? hi : lo;
        ret.m_hi = (n & 64) ? 0 : hi;

        return ret;
    }

    MAKE_BINARY_OP_HELPERS_uint64_t(>> );

    __device__ __host__ uint128_t& operator<<=(uint64_t n)
    {
        const uint64_t hi = __shiftleft128(m_lo, m_hi, (uint8_t)n);
        const uint64_t lo = m_lo << (n & 63ULL);

        m_hi = (n & 64) ? lo : hi;
        m_lo = (n & 64) ? 0 : lo;

        return *this;
    }

    friend __device__ __host__ uint128_t operator<<(const uint128_t& x, uint64_t n)
    {
        uint128_t ret;

        const uint64_t hi = __shiftleft128(x.m_lo, x.m_hi, (uint8_t)n);
        const uint64_t lo = x.m_lo << (n & 63ULL);

        ret.m_hi = (n & 64) ? lo : hi;
        ret.m_lo = (n & 64) ? 0 : lo;

        return ret;
    }

    MAKE_BINARY_OP_HELPERS_uint64_t(<< );

    friend __device__ __host__ uint128_t operator~(const uint128_t& x)
    {
        return uint128_t(~x.m_hi, ~x.m_lo);
    }

    friend __device__ __host__ uint128_t operator+(const uint128_t& x)
    {
        return x;
    }

    friend __device__ __host__ uint128_t operator-(const uint128_t& x)
    {
        uint128_t ret;
        _subborrow_u64(_subborrow_u64(0, 0, x.m_lo, &ret.m_lo), 0, x.m_hi, &ret.m_hi);
        return ret;
    }

    __device__ __host__ uint128_t& operator++()
    {
        operator+=(1);
        return *this;
    }

    __device__ __host__ uint128_t operator++(int)
    {
        const uint128_t x = *this;
        operator++();
        return x;
    }

    __device__ __host__ uint128_t& operator--()
    {
        operator-=(1);
        return *this;
    }

    __device__ __host__ uint128_t operator--(int)
    {
        const uint128_t x = *this;
        operator--();
        return x;
    }

    friend __device__ __host__ bool operator<(const uint128_t& x, const uint128_t& y)
    {
        uint64_t unusedLo, unusedHi;
        return _subborrow_u64(_subborrow_u64(0, x.m_lo, y.m_lo, &unusedLo), x.m_hi, y.m_hi, &unusedHi);
    }
    MAKE_BINARY_OP_HELPERS(< );
    MAKE_BINARY_OP_HELPERS_FLOAT(< );

    friend __device__ __host__ bool operator>(const uint128_t& x, const uint128_t& y) { return y < x; }
    MAKE_BINARY_OP_HELPERS(> );
    MAKE_BINARY_OP_HELPERS_FLOAT(> );

    friend __device__ __host__ bool operator<=(const uint128_t& x, const uint128_t& y) { return !(x > y); }
    MAKE_BINARY_OP_HELPERS(<= );
    MAKE_BINARY_OP_HELPERS_FLOAT(<= );

    friend __device__ __host__ bool operator>=(const uint128_t& x, const uint128_t& y) { return !(x < y); }
    MAKE_BINARY_OP_HELPERS(>= );
    MAKE_BINARY_OP_HELPERS_FLOAT(>= );

    friend __device__ __host__ bool operator==(const uint128_t& x, const uint128_t& y)
    {
        return !((x.m_hi ^ y.m_hi) | (x.m_lo ^ y.m_lo));
    }
    MAKE_BINARY_OP_HELPERS(== );
    MAKE_BINARY_OP_HELPERS_FLOAT(== );

    friend __device__ __host__ bool operator!=(const uint128_t& x, const uint128_t& y) { return !(x == y); }
    MAKE_BINARY_OP_HELPERS(!= );
    MAKE_BINARY_OP_HELPERS_FLOAT(!= );

    __device__ __host__ explicit operator bool() const { return m_hi | m_lo; }

    __device__ __host__ operator uint8_t () const { return (uint8_t)m_lo; }
    __device__ __host__ operator uint16_t() const { return (uint16_t)m_lo; }
    __device__ __host__ operator uint32_t() const { return (uint32_t)m_lo; }
    __device__ __host__ operator uint64_t() const { return (uint64_t)m_lo; }

    __device__ __host__ operator int8_t () const { return (int8_t)m_lo; }
    __device__ __host__ operator int16_t() const { return (int16_t)m_lo; }
    __device__ __host__ operator int32_t() const { return (int32_t)m_lo; }
    __device__ __host__ operator int64_t() const { return (int64_t)m_lo; }

    __device__ __host__ operator char() const { return (char)m_lo; }

    __device__ __host__ operator float() const;
    __device__ __host__ operator double() const;

    __device__ __host__ void ToString(char* buf, uint64_t base = 10) const;

};

#undef MAKE_BINARY_OP_HELPERS
#undef MAKE_BINARY_OP_HELPERS_FLOAT
#undef MAKE_BINARY_OP_HELPERS_uint64_t

__host__ std::ostream& operator<<(std::ostream& os, const uint128_t& x);


__device__ __host__ static inline bool FitsHardwareDivL(uint64_t nHi, uint64_t nLo, uint64_t d)
{
    return !(nHi | (d >> 32)) && nLo < (d << 32);
}

__device__ __host__ static inline bool IsPow2(uint64_t hi, uint64_t lo)
{
    const uint64_t T = hi | lo;
    return !((hi & lo) | (T & (T - 1)));
}

__host__ static inline uint64_t HardwareDivL(uint64_t n, uint64_t d, uint64_t& rem)
{
    uint32_t qLo = uint128_t::_udiv64(n, (uint64_t)d);
    rem = (uint64_t)(n % d);
    return qLo;
}

__host__ static inline uint64_t HardwareDivQ(uint64_t nHi, uint64_t nLo, uint64_t d, uint64_t& rem)
{
    return uint128_t::_udiv128(nHi, nLo, d, &rem);
}

__host__ static inline uint64_t CountTrailingZeros(uint64_t hi, uint64_t lo)
{
    const uint64_t nLo = _tzcnt_u64(lo);
    const uint64_t nHi = 64ULL + _tzcnt_u64(hi);
    return lo ? nLo : nHi;
}

__host__ static inline uint64_t CountLeadingZeros(uint64_t hi, uint64_t lo)
{
    const uint64_t nLo = 64ULL + _lzcnt_u64(lo);
    const uint64_t nHi = _lzcnt_u64(hi);
    return hi ? nHi : nLo;
}

__host__ static inline uint128_t MaskBitsBelow(uint64_t hi, uint64_t lo, uint64_t n)
{
    return uint128_t(_bzhi_u64(hi, uint32_t(n < 64 ? 0 : n - 64)), _bzhi_u64(lo, uint32_t(n)));
}


__device__ __host__ __inline__  uint128_t DivMod(uint128_t N, uint128_t D, uint128_t& rem)
{
    if (D > N)
    {
        rem = N;
        return 0;
    }

    uint64_t nHi = N.m_hi;
    uint64_t nLo = N.m_lo;
    uint64_t dHi = D.m_hi;
    uint64_t dLo = D.m_lo;
#ifdef __CUDA_ARCH__
    if ((dHi == 0) && ((dLo & (dLo - 1)) == 0))
    {
        int n = __ffsll(dLo) - 1;
        rem = uint128_t(nHi << (64 - n) | nLo >> n, nLo & ((1ULL << n) - 1));
        return N >> n;
    }

    if (!dHi)
    {
        if (nHi < dLo)
        {
            uint64_t Q = nLo / dLo;
            uint64_t remLo = nLo % dLo;
            rem = uint128_t(0, remLo);
            return uint128_t(0, Q);
        }

        uint128_t n = uint128_t(nHi, nLo);
        uint128_t q = n / dLo;
        rem = n % dLo;
        return q;
    }

    uint64_t n = _lzcnt_u64(dHi) - _lzcnt_u64(nHi);

    uint128_t shiftedD = D << n;
    dHi = shiftedD.m_hi;
    dLo = shiftedD.m_lo;

    uint64_t Q = 0;
    ++n;

    do
    {
        uint128_t t = nHi >= dHi ? uint128_t(nHi, nLo) - uint128_t(dHi, dLo) : uint128_t(nHi, nLo);
        bool carry = nHi >= dHi;
        nHi = t.m_hi;
        nLo = t.m_lo;
        Q = (Q << 1) | carry;
        shiftedD >>= 1;
        dHi = shiftedD.m_hi;
        dLo = shiftedD.m_lo;
    } while (--n);

    rem = uint128_t(nHi, nLo);
    return Q;
#else
    if (IsPow2(dHi, dLo))
    {
        const uint64_t n = CountTrailingZeros(dHi, dLo);
        rem = MaskBitsBelow(nHi, nLo, n);
        return N >> n;
    }

    if (!dHi)
    {
        if (nHi < dLo)
        {
            uint64_t remLo;
            uint64_t Q;
            if (FitsHardwareDivL(nHi, nLo, dLo))
                Q = HardwareDivL(nLo, dLo, remLo);
            else
                Q = HardwareDivQ(nHi, nLo, dLo, remLo);
            rem = remLo;
            return Q;
        }

        uint64_t remLo;
        const uint64_t qHi = HardwareDivQ(0, nHi, dLo, remLo);
        const uint64_t qLo = HardwareDivQ(remLo, nLo, dLo, remLo);
        rem = remLo;
        return uint128_t(qHi, qLo);
    }

    uint64_t n = _lzcnt_u64(dHi) - _lzcnt_u64(nHi);

    dHi = uint128_t::__shiftleft128(dLo, dHi, uint8_t(n));
    dLo <<= n;

    uint64_t Q = 0;
    ++n;

    do
    {
        uint64_t tLo, tHi;
        unsigned char carry = _subborrow_u64(_subborrow_u64(0, nLo, dLo, &tLo), nHi, dHi, &tHi);
        nLo = !carry ? tLo : nLo;
        nHi = !carry ? tHi : nHi;
        Q = (Q << 1) + !carry;
        dLo = uint128_t::__shiftright128(dLo, dHi, 1);
        dHi >>= 1;
    } while (--n);

    rem = uint128_t(nHi, nLo);
    return Q;
#endif
}

#endif // UINT128_T_H
