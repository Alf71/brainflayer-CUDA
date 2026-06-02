#include "Kernel.cuh"

__device__ static bool mask_candidate_from_spec(const BrainMaskSpec* __restrict__ spec, uint64_t ordinal, char* out)
{
    if (spec == nullptr || spec->len == 0u || spec->len > BRAIN_MASK_MAX_LEN) {
        return false;
    }
    for (int pos = (int)spec->len - 1; pos >= 0; --pos) {
        const uint32_t clen = spec->charset_len[pos];
        if (clen == 0u) {
            return false;
        }
        const uint32_t digit = (uint32_t)(ordinal % (uint64_t)clen);
        ordinal /= (uint64_t)clen;
        out[pos] = (char)spec->chars[(uint32_t)spec->charset_offset[pos] + digit];
    }
    return true;
}

__global__ void buildMaskBatch(const BrainMaskSpec* __restrict__ spec, uint64_t candidate_start, char* __restrict__ lines, uint32_t* __restrict__ indexes, uint32_t count)
{
    if (spec == nullptr || lines == nullptr || indexes == nullptr) {
        return;
    }
    const uint32_t len = spec->len;
    if (len == 0u || len > BRAIN_MASK_MAX_LEN) {
        return;
    }
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; tid < count; tid += stride) {
        char* dst = lines + tid * (uint64_t)len;
        mask_candidate_from_spec(spec, candidate_start + tid, dst);
        indexes[tid] = (uint32_t)((tid + 1u) * len);
    }
}
