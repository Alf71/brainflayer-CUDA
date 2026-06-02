#include "stdint.h"



/* endian */

__device__  inline void U32TO8_LE(unsigned char* p, const uint32_t v);

__device__  inline uint32_t U8TO32_LE(const unsigned char* p);

#include <stdlib.h>
#include <string.h>


