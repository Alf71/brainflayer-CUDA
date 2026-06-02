#include "Kernel.cuh"

#define HASH_LEN 32
#define PUBKEY_LEN 32
#define ROOT_HEADER_SIZE 39
#define DATA_HEADER_SIZE 6
#define SUBWALLET_SIZE 4
#define DATA_TAIL_SIZE 1

__device__ const uint8_t highload_v1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x04, 0x00, 0x00,
	0x0A, 0xB1, 0xFF, 0x93, 0xA9, 0xE7, 0x9C, 0x0D,
	0xEF, 0xF4, 0x40, 0x5D, 0x8D, 0xAD, 0x6B, 0x5A,
	0xC9, 0xF8, 0xC0, 0x1B, 0x2F, 0x1E, 0xBC, 0xB4,
	0x25, 0x9F, 0xEB, 0x9E, 0x98, 0x38, 0x00, 0x99

};
__device__ const  uint8_t highload_v2_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x04, 0x00, 0x00,
	0x94, 0x94, 0xD1, 0xCC, 0x8E, 0xDF, 0x12, 0xF0,
	0x56, 0x71, 0xA1, 0xA9, 0xBA, 0x09, 0x92, 0x10,
	0x96, 0xEB, 0x50, 0x81, 0x1E, 0x19, 0x24, 0xEC,
	0x65, 0xC3, 0xC6, 0x29, 0xFB, 0xB8, 0x08, 0x12

};
__device__ const  uint8_t highload_v3_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x06, 0x00, 0x00,
	0x11, 0xAC, 0xAD, 0x79, 0x55, 0x84, 0x40, 0x90,
	0xF2, 0x83, 0xBF, 0x23, 0x8B, 0xC1, 0x44, 0x98,
	0x71, 0xF7, 0x83, 0xE7, 0xCC, 0x09, 0x79, 0x40,
	0x8D, 0x3F, 0x48, 0x59, 0x48, 0x3E, 0x85, 0x25

};
__device__ const uint8_t v5r1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x06, 0x00, 0x00,
	0x20, 0x83, 0x4B, 0x7B, 0x72, 0xB1, 0x12, 0x14,
	0x7E, 0x1B, 0x2F, 0xB4, 0x57, 0xB8, 0x4E, 0x74,
	0xD1, 0xA3, 0x0F, 0x04, 0xF7, 0x37, 0xD4, 0xF6,
	0x2A, 0x66, 0x8E, 0x95, 0x52, 0xD2, 0xB7, 0x2F
};


__device__ const  uint8_t v4r2_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x07, 0x00, 0x00,
	0xfe, 0xb5, 0xff, 0x68, 0x20, 0xe2, 0xff, 0x0d,
	0x94, 0x83, 0xe7, 0xe0, 0xd6, 0x2c, 0x81, 0x7d,
	0x84, 0x67, 0x89, 0xfb, 0x4a, 0xe5, 0x80, 0xc8,
	0x78, 0x86, 0x6d, 0x95, 0x9d, 0xab, 0xd5, 0xc0
};

__device__ const  uint8_t v4r1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x07, 0x00, 0x00,
	0x64, 0xdd, 0x54, 0x80, 0x55, 0x22, 0xc5, 0xbe, 0x8a, 0x9d, 0xb5, 0x9c, 0xea, 0x01, 0x05, 0xcc,
0xf0, 0xd0, 0x87, 0x86, 0xca, 0x79, 0xbe, 0xb8, 0xcb, 0x79, 0xe8, 0x80, 0xa8, 0xd7, 0x32, 0x2d
};

__device__ const  uint8_t v4_data_header[] = {
	0x00, 0x51, 0x00, 0x00, 0x00, 0x00
};

__device__ const  uint8_t data_tail[] = { 0x40 };


__device__ const  uint8_t v3r2_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0x84, 0xda, 0xfa, 0x44, 0x9f, 0x98, 0xa6, 0x98, 0x77, 0x89, 0xba, 0x23, 0x23, 0x58, 0x7,  0x2b, 0xc0, 0xf7, 0x6d, 0xc4, 0x52, 0x40, 0x2, 0xa5, 0xd0, 0x91, 0x8b, 0x9a, 0x75, 0xd2, 0xd5, 0x99 };

__device__ const  uint8_t v3r1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0xb6, 0x10, 0x41, 0xa5, 0x8a, 0x79, 0x80, 0xb9, 0x46, 0xe8, 0xfb, 0x9e, 0x19, 0x8e, 0x3c, 0x90,
0x4d, 0x24, 0x79, 0x9f, 0xfa, 0x36, 0x57, 0x4e, 0xa4, 0x25, 0x1c, 0x41, 0xa5, 0x66, 0xf5, 0x81 };



__device__ const  uint8_t v2r1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0x5C, 0x9A, 0x5E, 0x68, 0xC1, 0x08, 0xE1, 0x87,
	0x21, 0xA0, 0x7C, 0x42, 0xF9, 0x95, 0x6B, 0xFB,
	0x39, 0xAD, 0x77, 0xEC, 0x6D, 0x62, 0x4B, 0x60,
	0xC5, 0x76, 0xEC, 0x88, 0xEE, 0xE6, 0x53, 0x29
};


__device__ const  uint8_t v2r2_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0xFE, 0x95, 0x30, 0xD3, 0x24, 0x38, 0x53, 0x08,
	0x3E, 0xF2, 0xEF, 0x0B, 0x4C, 0x29, 0x08, 0xC0,
	0xAB, 0xF6, 0xFA, 0x1C, 0x31, 0xEA, 0x24, 0x3A,
	0xAC, 0xAA, 0x5B, 0xF8, 0xC7, 0xD7, 0x53, 0xF1
};

__device__ const  uint8_t v1r1_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0xA0, 0xCF, 0xC2, 0xC4, 0x8A, 0xEE, 0x16, 0xA2,
	0x71, 0xF2, 0xCF, 0xC0, 0xB7, 0x38, 0x2D, 0x81,
	0x75, 0x6C, 0xEC, 0xB1, 0x01, 0x7D, 0x07, 0x7F,
	0xAA, 0xAB, 0x3B, 0xB6, 0x02, 0xF6, 0x86, 0x8C
};

__device__  const uint8_t v1r2_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0xD4, 0x90, 0x2F, 0xCC, 0x9F, 0xAD, 0x74, 0x69,
	0x8F, 0xA8, 0xE3, 0x53, 0x22, 0x0A, 0x68, 0xDA,
	0x0D, 0xCF, 0x72, 0xE3, 0x2B, 0xCB, 0x2E, 0xB9,
	0xEE, 0x04, 0x21, 0x7C, 0x17, 0xD3, 0x06, 0x2C
};

__device__ const  uint8_t v1r3_root_header[] = {
	0x02, 0x01, 0x34, 0x00, 0x00, 0x00, 0x00,
	0x58, 0x7C, 0xC7, 0x89, 0xEF, 0xF1, 0xC8, 0x4F,
	0x46, 0xEC, 0x37, 0x97, 0xE4, 0x5F, 0xC8, 0x09,
	0xA1, 0x4F, 0xF5, 0xAE, 0x24, 0xF1, 0xE0, 0xC7,
	0xA6, 0xA9, 0x9C, 0xC9, 0xDC, 0x90, 0x61, 0xFF
};



__device__ const  uint8_t v2_data_header[] = {
	0x00,
	0x48,  // Cell header
	0x00,
	0x00,
	0x00,
	0x00,  // Seqno
};

__device__ const  uint8_t v1_data_header[] = {
	0x00,
	0x48,  // Cell header
	0x00,
	0x00,
	0x00,
	0x00,  // Seqno
};

__device__  const uint8_t v3_data_header[] = {
	0x00,
	0x50,  // Cell header
	0x00,
	0x00,
	0x00,
	0x00,  // Seqno
};

__device__ const  uint8_t highload_v1_data_header[] = {
	0x00,
	0x50,  // Cell header
	0x00,
	0x00,
	0x00,
	0x00,  // Seqno
};
__device__  const uint8_t highload_v2_data_header[] = {
	0x00,
	0x59,
	0x29,
	0xA9,
	0xA3,
	0x17,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00
};

__device__  const uint8_t highload_v3_data_header[] = { 0x00, 0x5e };

__device__  const uint8_t highload_v3_data_footer[] = { 0x00, 0x00, 0x00, 0xef, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 };

__device__ uint32_t subwallet_id = 698983191;
__device__ uint32_t subwallet_id_v5 = 2147483409;


// Compares two zero-terminated strings on device.
__device__ __forceinline__ bool eq(const char* a, const char* b)
{
	for (; *a || *b; ++a, ++b)
		if (*a != *b) return false;
	return true;
}

// Appends a 32-bit value bit-by-bit in big-endian order to a packed bitstream.
__device__ __forceinline__ void PUT_BITS_BE32(uint8_t* buf, size_t* len, uint8_t* pcur, int* pbitpos, uint32_t v)
{
	uint8_t* data = buf;
	size_t* ldata = len;
	uint8_t* curp = pcur;
	int* bpos = pbitpos;
	int i;
	for (i = 31; i >= 0; --i) {
		uint8_t bit = (uint8_t)((v >> i) & 1u);
		*curp |= (uint8_t)(bit << (7 - *bpos));
		(*bpos)++;
		if (*bpos == 8) {
			data[(*ldata)++] = *curp;
			*curp = 0;
			*bpos = 0;
		}
	}
}

// Appends raw bytes bit-by-bit in big-endian order to a packed bitstream.
__device__ __forceinline__ void PUT_BYTES_BITS_BE(uint8_t* buf, size_t* len, uint8_t* pcur, int* pbitpos, const uint8_t* src, size_t n)
{
	uint8_t* data = buf;
	size_t* ldata = len;
	uint8_t* curp = pcur;
	int* bpos = pbitpos;
	size_t i;
	for (i = 0; i < n; ++i) {
		uint8_t byte = src[i];
		int b;
		for (b = 7; b >= 0; --b) {
			uint8_t bit = (uint8_t)((byte >> b) & 1u);
			*curp |= (uint8_t)(bit << (7 - *bpos));
			(*bpos)++;
			if (*bpos == 8) {
				data[(*ldata)++] = *curp;
				*curp = 0;
				*bpos = 0;
			}
		}
	}
}

// Builds TON wallet hash payload for specific wallet type from a public key.
__device__  bool pubkey_to_hash_ton(const uint8_t* public_key, const
	char* type,
	uint8_t* out,
	size_t out_len ) {
	uint8_t inner[HASH_LEN] = { 0 };
	uint8_t subwallet_buf[4];
	subwallet_buf[0] = (subwallet_id >> 24) & 0xff;
	subwallet_buf[1] = (subwallet_id >> 16) & 0xff;
	subwallet_buf[2] = (subwallet_id >> 8) & 0xff;
	subwallet_buf[3] = subwallet_id & 0xff;
	if (eq(type, "v1r1") || eq(type, "v1r2") || eq(type, "v1r3"))
	{
		uint8_t data1[128] = { 0 };
		memcpy(data1, v1_data_header, DATA_HEADER_SIZE);
		memcpy(data1 + DATA_HEADER_SIZE, public_key, PUBKEY_LEN);
		sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + PUBKEY_LEN, (uint32_t*)inner);

		const uint8_t* data2_header = nullptr;
		if (eq(type, "v1r1"))
		{
			data2_header = v1r1_root_header;
		}
		else if (eq(type, "v1r2"))
		{
			data2_header = v1r2_root_header;
		}
		else if (eq(type, "v1r3"))
		{
			data2_header = v1r3_root_header;
		}

		uint8_t data2[128] = { 0 };
		memcpy(data2, data2_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);
		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
	}
	else if (eq(type, "v2r1") || eq(type, "v2r2"))
	{
		uint8_t data1[128] = { 0 };
		memcpy(data1, v2_data_header, DATA_HEADER_SIZE);
		memcpy(data1 + DATA_HEADER_SIZE, public_key, PUBKEY_LEN);
		sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + PUBKEY_LEN, (uint32_t*)inner);
		const uint8_t* root_header = nullptr;
		if (eq(type, "v2r1"))
		{
			root_header = v2r1_root_header;
		}
		else if (eq(type, "v2r2"))
		{
			root_header = v2r2_root_header;
		}
		uint8_t data2[128] = { 0 };
		memcpy(data2, root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);

		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
	}
	else if (eq(type, "v3r1" )|| eq(type, "v3r2"))
	{
		uint8_t data1[128] = { 0 };
		size_t offset = 0;
		memcpy(data1 + offset, v3_data_header, DATA_HEADER_SIZE);
		offset += DATA_HEADER_SIZE;
		memcpy(data1 + offset, subwallet_buf, SUBWALLET_SIZE);
		offset += SUBWALLET_SIZE;
		memcpy(data1 + offset, public_key, PUBKEY_LEN);

		sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + SUBWALLET_SIZE + PUBKEY_LEN, (uint32_t*)inner);
		const uint8_t* root_header = nullptr;

		if (eq(type, "v3r1"))
		{
			root_header = v3r1_root_header;
		}
		else if (eq(type, "v3r2"))
		{
			root_header = v3r2_root_header;
		}

		uint8_t data2[128] = { 0 };
		memcpy(data2, root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);
		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
	}
	else if (eq(type, "v4r1") || eq(type, "v4r2"))
	{

		uint8_t data1[128] = { 0 };
		size_t offset = 0;
		memcpy(data1 + offset, v4_data_header, DATA_HEADER_SIZE);
		offset += DATA_HEADER_SIZE;
		memcpy(data1 + offset, subwallet_buf, SUBWALLET_SIZE);
		offset += SUBWALLET_SIZE;
		memcpy(data1 + offset, public_key, PUBKEY_LEN);
		offset += PUBKEY_LEN;
		memcpy(data1 + offset, data_tail, DATA_TAIL_SIZE);

		sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + SUBWALLET_SIZE + PUBKEY_LEN + DATA_TAIL_SIZE, (uint32_t*)inner);

		const uint8_t* root_header = nullptr;

		if (eq(type, "v4r1"))
		{
			root_header = v4r1_root_header;
		}
		else if (eq(type, "v4r2"))
		{
			root_header = v4r2_root_header;
		}

		uint8_t data2[128] = { 0 };
		memcpy(data2, root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);

		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
	}
	else if (eq(type, "v5r1"))
	{
		uint8_t data1[128] = { 0 };
		size_t data_len = 0;
		uint8_t cur = 0;
		int bitpos = 0;


#define FLUSH_BYTE() do { data1[data_len++] = cur; cur = 0; bitpos = 0; } while (0)
#define PUT_BIT(b) do { \
        cur |= ((uint8_t)((b) & 1)) << (7 - bitpos); \
        bitpos++; \
        if (bitpos == 8) FLUSH_BYTE(); \
    } while (0)

		data1[data_len++] = 0x00;
		data1[data_len++] = 0x51;
		PUT_BIT(1);
		{
			uint32_t seqno = 0;
			PUT_BITS_BE32(data1, &data_len, &cur, &bitpos, seqno);
		}
		{
			uint32_t sid = subwallet_id_v5;
			PUT_BITS_BE32(data1, &data_len, &cur, &bitpos, sid);
		}
		PUT_BYTES_BITS_BE(data1, &data_len, &cur, &bitpos, public_key, PUBKEY_LEN);

		PUT_BIT(0);

		PUT_BIT(1);

		if (bitpos != 0) {
			FLUSH_BYTE();
		}
		sha256_d((uint32_t*)data1, (uint32_t)data_len, (uint32_t*)inner);
		uint8_t data2[ROOT_HEADER_SIZE + HASH_LEN] = { 0 };
		memcpy(data2, v5r1_root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);
		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
#undef FLUSH_BYTE
#undef PUT_BIT
	}
	else if (eq(type, "hv1") || eq(type, "hv2"))
	{
		uint8_t data1[128] = { 0 };
		if (eq(type, "hv1"))
		{
			size_t offset = 0;
			memcpy(data1 + offset, highload_v1_data_header, DATA_HEADER_SIZE);
			offset += DATA_HEADER_SIZE;
			memcpy(data1 + offset, subwallet_buf, SUBWALLET_SIZE);
			offset += SUBWALLET_SIZE;
			memcpy(data1 + offset, public_key, PUBKEY_LEN);

			sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + SUBWALLET_SIZE + PUBKEY_LEN + DATA_TAIL_SIZE - 1, (uint32_t*)inner);
		}
		else if (eq(type, "hv2"))
		{
			size_t offset = 0;
			memcpy(data1 + offset, highload_v2_data_header, DATA_HEADER_SIZE + SUBWALLET_SIZE + 4);
			offset += DATA_HEADER_SIZE + SUBWALLET_SIZE + 4;
			memcpy(data1 + offset, public_key, PUBKEY_LEN);
			offset += PUBKEY_LEN;
			memcpy(data1 + offset, data_tail, DATA_TAIL_SIZE);
			sha256_d((uint32_t*)data1, DATA_HEADER_SIZE + SUBWALLET_SIZE + PUBKEY_LEN + 4 + DATA_TAIL_SIZE, (uint32_t*)inner);

		}
		const uint8_t* root_header = nullptr;
		if (eq(type, "hv1"))
		{
			root_header = highload_v1_root_header;
		}
		else if (eq(type, "hv2"))
		{
			root_header = highload_v2_root_header;
		}

		uint8_t data2[128] = { 0 };
		memcpy(data2, root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);

		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);
	}
	else if (eq(type, "hv3"))
	{

		uint8_t data1[128] = { 0 };

		size_t offset = 0;
		memcpy(data1 + offset, highload_v3_data_header, sizeof(highload_v3_data_header));
		offset += sizeof(highload_v3_data_header);
		memcpy(data1 + offset, public_key, PUBKEY_LEN);
		offset += PUBKEY_LEN;
		memcpy(data1 + offset, highload_v3_data_footer, sizeof(highload_v3_data_footer));

		sha256_d((uint32_t*)data1, sizeof(highload_v3_data_header) + PUBKEY_LEN + sizeof(highload_v3_data_footer), (uint32_t*)inner);

		const uint8_t* root_header = nullptr;
		root_header = highload_v3_root_header;


		uint8_t data2[128] = { 0 };
		memcpy(data2, root_header, ROOT_HEADER_SIZE);
		memcpy(data2 + ROOT_HEADER_SIZE, inner, HASH_LEN);

		sha256_d((uint32_t*)data2, ROOT_HEADER_SIZE + HASH_LEN, (uint32_t*)out);



	}

	return true;
}
