/*
    conversions
*/
#pragma once
#include "modm-donna-32bit.h"
__device__   void
ge25519_p1p1_to_partial(ge25519* r, const ge25519_p1p1* p);

__device__   void
ge25519_p1p1_to_full(ge25519* r, const ge25519_p1p1* p);

__device__  void
ge25519_full_to_pniels(ge25519_pniels* p, const ge25519* r);

/*
    adding & doubling
*/

__device__  void
ge25519_add_p1p1(ge25519_p1p1* r, const ge25519* p, const ge25519* q);


__device__  void
ge25519_double_p1p1(ge25519_p1p1* r, const ge25519* p);

__device__  void
ge25519_nielsadd2_p1p1(ge25519_p1p1* r, const ge25519* p, const ge25519_niels* q, unsigned char signbit);

__device__  void
ge25519_pnielsadd_p1p1(ge25519_p1p1* r, const ge25519* p, const ge25519_pniels* q, unsigned char signbit);

__device__  void
ge25519_double_partial(ge25519* r, const ge25519* p);

__device__  void
ge25519_double(ge25519* r, const ge25519* p);

__device__  void
ge25519_add(ge25519* r, const ge25519* p, const ge25519* q);

__device__  void
ge25519_nielsadd2(ge25519* r, const ge25519_niels* q);

__device__  void
ge25519_pnielsadd(ge25519_pniels* r, const ge25519* p, const ge25519_pniels* q);


/*
    pack & unpack
*/

__device__  void
ge25519_pack(unsigned char r[32], const ge25519* p);

__device__  int
ge25519_unpack_negative_vartime(ge25519* r, const unsigned char p[32]);


/* computes [s1]p1 + [s2]basepoint */
__device__  void
ge25519_double_scalarmult_vartime(ge25519* r, const ge25519* p1, const bignum256modm s1, const bignum256modm s2);



#if !defined(HAVE_GE25519_SCALARMULT_BASE_CHOOSE_NIELS)

__device__  uint32_t
ge25519_windowb_equal(uint32_t b, uint32_t c);

__device__  void
ge25519_scalarmult_base_choose_niels(ge25519_niels* t, const uint8_t table[256][96], uint32_t pos, signed char b);

#endif /* HAVE_GE25519_SCALARMULT_BASE_CHOOSE_NIELS */


/* computes [s]basepoint */
__device__  void
ge25519_scalarmult_base_niels(ge25519* r, const uint8_t basepoint_table[256][96], const bignum256modm s);
