#pragma once
#include <cstdint>
/* multiples of the base point in packed {ysubx, xaddy, t2d} form */
extern __device__  const uint8_t __align__(16) ge25519_niels_base_multiples[256][96];
