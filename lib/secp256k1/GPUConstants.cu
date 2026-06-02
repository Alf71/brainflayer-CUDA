// Single compilation unit for GPUMath/GPUGroup device constants.
// Prevents multiple definitions and keeps constant memory under 64 KB limit.
#define GPU_CONSTANTS_DEFINITIONS
#include "secp256k1_common.cuh"
#include "GPUMath.cuh"
#include "GPUGroup.cuh"
