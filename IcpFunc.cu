#include "Kernel.cuh"


// DER prefix used by ICP for Ed25519 public keys.
__device__ __constant__ uint8_t DER_ED25519_PREFIX[] = {
  0x30,0x2a,0x30,0x05,0x06,0x03,0x2b,0x65,0x70,0x03,0x21,0x00
};

// DER prefix used by ICP for secp256k1 public keys.
__device__ __constant__ uint8_t DER_SECP256K1_PREFIX[] = {
0x30,0x56,0x30,0x10,0x06,0x07,0x2a,0x86,0x48,0xce,0x3d,0x02,0x01,0x06,0x05,0x2b,0x81,0x04,0x00,0x0a,0x03,0x42,0x00,0x04
};

// Fixed SHA-224 output length for ICP principals.
__device__ __constant__ int ICP_SHA224_LEN = 28;

// Principal type tag for self-authenticating principals.
__device__ __constant__ uint8_t ICP_PRINCIPAL_TYPE_SELF_AUTH = 0x02;

// Stores a 32-bit integer in big-endian byte order.
__device__ __inline__ void u32_to_be(uint32_t x, uint8_t out[4]) {
	out[0] = (uint8_t)((x >> 24) & 0xFF);
	out[1] = (uint8_t)((x >> 16) & 0xFF);
	out[2] = (uint8_t)((x >> 8) & 0xFF);
	out[3] = (uint8_t)(x & 0xFF);
}

// Builds a 29-byte ICP principal from an Ed25519 public key.
__device__ int icp_principal_from_ed25519(const uint8_t pub_ed25519[32], uint8_t out_principal[29]) {
	uint8_t buf[sizeof(DER_ED25519_PREFIX) + 32];
	memcpy(buf, DER_ED25519_PREFIX, sizeof(DER_ED25519_PREFIX));
	memcpy(buf + sizeof(DER_ED25519_PREFIX), pub_ed25519, 32);
	SHA224(buf, sizeof(buf), out_principal);              // 28 bytes
	out_principal[ICP_SHA224_LEN] = ICP_PRINCIPAL_TYPE_SELF_AUTH; // 0x02
	return 0;
}

// Builds a 29-byte ICP principal from an uncompressed secp256k1 public key.
__device__ int icp_principal_from_secp256k1(const uint8_t pub_secp256k1[65], uint8_t out_principal[29]) {
	const uint8_t* xy64 = pub_secp256k1 + 1; // X||Y
	uint8_t buf[sizeof(DER_SECP256K1_PREFIX) + 64];
	memcpy(buf, DER_SECP256K1_PREFIX, sizeof(DER_SECP256K1_PREFIX));
	memcpy(buf + sizeof(DER_SECP256K1_PREFIX), xy64, 64);
	SHA224(buf, sizeof(buf), out_principal);              // 28 bytes
	out_principal[ICP_SHA224_LEN] = ICP_PRINCIPAL_TYPE_SELF_AUTH; // 0x02
	return 0;
}

// Builds a 32-byte ICP account identifier (CRC32 + SHA224 digest payload).
__device__ int icp_account_identifier(const uint8_t principal29[29],
	const uint8_t* subaccount32, // NULL
	uint8_t out_ai32[32]) {
	static const uint8_t DOMAIN_TAG_LEN = 0x0A; //  "account-id"
	static const char DOMAIN_TAG_STR[] = "account-id";
	uint8_t zero_sub[32];
	if (!subaccount32) {
		memset(zero_sub, 0, sizeof(zero_sub));
		subaccount32 = zero_sub;
	}

	// blob
	uint8_t blob[1 + sizeof(DOMAIN_TAG_STR) - 1 + 29 + 32];
	size_t off = 0;
	blob[off++] = DOMAIN_TAG_LEN;
	memcpy(blob + off, DOMAIN_TAG_STR, sizeof(DOMAIN_TAG_STR) - 1); off += (sizeof(DOMAIN_TAG_STR) - 1);
	memcpy(blob + off, principal29, 29); off += 29;
	memcpy(blob + off, subaccount32, 32); off += 32;

	// hash and CRC
	uint8_t h[28];
	SHA224(blob, off, h);
	uint32_t crc = crc32_ieee(h, ICP_SHA224_LEN);
	u32_to_be(crc, out_ai32);
	memcpy(out_ai32 + 4, h, ICP_SHA224_LEN); // 4 + 28 = 32
	return 0;
}
