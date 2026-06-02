#include "Kernel.cuh"

#include "lib/secp256k1/secp256k1.cuh"
#include "lib/secp256k1/secp256k1_batch_impl.cuh"

#include "sr25519-donna-32bit/sr25519.h"
#include "sr25519-donna-32bit/ed25519-donna/ed25519.h"

#include "big_int/big_int_device.cuh"

__global__ void workerBrain_seq(bool* isResult, bool* buffResult, const secp256k1_ge_storage* __restrict__ precPtr, const size_t precPitch, uint64_t round, uint8_t brain_mode, int mode, uint8_t* start_point_dev, int min_len, uint32_t iter) {
	const uint32_t tIx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t len = 0;
	__align__(16) unsigned char pubKeys[THREAD_STEPS_BRAIN * 65];
	__align__(16) unsigned char prvKeys[THREAD_STEPS_BRAIN * 32];
	__align__(16) unsigned char pubKeysED[THREAD_STEPS_BRAIN * 32];

	uint32_t hash160[8];
	uint64_t starter = THREAD_STEPS_BRAIN * tIx * SeqStep;
	char toHashBuffer[THREAD_STEPS_BRAIN][256];
	int privKeyIx = 0;

	const bool secp256_dev = secp256_d;
	const bool ed25519_dev = ed25519_d;


	const bool compressed = compressed_dev;
	const bool uncompressed = uncompressed_dev;
	const bool segwit = segwit_dev;
	const bool taproot = taproot_dev;
	const bool ethereum = ethereum_dev;
	const bool xpoint = xpoint_dev;
	const bool solana = solana_dev;
	const bool ton = ton_dev;
	const bool ton_all = ton_all_dev;
	const bool dot = dot_dev;
	const bool aptos = aptos_dev;
	const bool sui = sui_dev;
	const bool xrp = xrp_dev;
	const bool iota = iota_dev;
	const bool icp = icp_dev;
	const bool fil = fil_dev;
	const bool xtz = xtz_dev;

	uint32_t ITERATION = iter;

	__align__(16) uint32_t c[64];
	loadBE2048toWords(start_point_dev, c);

	while (privKeyIx < THREAD_STEPS_BRAIN) {
		__align__(16) uint32_t r[64];

		uint64_t idx = starter + (uint64_t)privKeyIx * SeqStep;

		if (mode == 1)
		{
			add64to2048_device(c, idx, r);
		}
		else
		{
			sub64from2048_device(c, idx, r);
		}
		len = sizeBE2048_device(r);

		if (len < min_len)
		{
			len = min_len;
		}

		uint8_t toHash[256];

		storeWordsToBE2048(r, toHash);
		for (int j = 0; j < len; j++) {
			toHashBuffer[privKeyIx][j] = toHash[256 - len + j];
		}

		int pk4 = privKeyIx << 5;

		switch (brain_mode)
		{
		case 1:
			sha3_256(toHashBuffer[privKeyIx], len, &prvKeys[pk4]);
			for (int i = 1; i < ITERATION; i++)
			{
				sha3_256((char*)&prvKeys[pk4], 32, &prvKeys[pk4]);
			}
			break;
		case 2:
			keccak(toHashBuffer[privKeyIx], len, &prvKeys[pk4], 32);
			for (int i = 1; i < ITERATION; i++)
			{
				keccak((char*)&prvKeys[pk4], 32, &prvKeys[pk4], 32);
			}
			break;
		case 3:
			Blake2b_256((uint8_t*)&toHashBuffer[privKeyIx], len, &prvKeys[pk4]);
			for (int i = 1; i < ITERATION; i++)
			{
				Blake2b_256(&prvKeys[pk4], 32, &prvKeys[pk4]);
			}
			break;
		default:
			SHA256((uint8_t*)&toHashBuffer[privKeyIx], len, &prvKeys[pk4]);
			for (int i = 1; i < ITERATION; i++)
			{
				SHA256(&prvKeys[pk4], 32, &prvKeys[pk4]);
			}
			break;
		}

		privKeyIx++;
	}

	if (round > 0) bump_all_keys(prvKeys, privKeyIx, round, false, NULL);

	for (uint64_t i = 0; i <= 2 * round; ++i)
	{
		int64_t current_round = (int64_t)i - (int64_t)round;

		if (secp256_dev)
		{
			if (i == 0)
			{
				secp256k1_ec_pubkey_create_serialized_batch_myunsafe(pubKeys, prvKeys, privKeyIx, precPtr, precPitch);
			}
			int keyLenSkip;
			int pkField = 0;

			__align__(4) uint8_t tap_hash[32 * THREAD_STEPS_BRAIN] = { 0 };
			if (taproot)
			{
				TweakTaproot_batch(tap_hash, pubKeys, privKeyIx, precPtr, precPitch);
			}

			for (pkField = 0; pkField < THREAD_STEPS_BRAIN && pkField < privKeyIx; pkField++) {

				if (uncompressed) {
					keyLenSkip = 65 * pkField;
					_GetHash160(pubKeys, keyLenSkip, (uint8_t*)hash160);
					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x01;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}

					}
				}
				if (compressed) {
					keyLenSkip = 65 * pkField;
					_GetHash160Comp(pubKeys, keyLenSkip, (uint8_t*)hash160);
					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x02;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (segwit) {
					keyLenSkip = 65 * pkField;
					if (!compressed) _GetHash160Comp(pubKeys, keyLenSkip, (uint8_t*)hash160);
					_GetHash160P2SHCompFromHash(hash160, hash160);

					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x03;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (taproot)
				{
					uint8_t* TaprootHash = &tap_hash[pkField * 32];

					_GetRMD160((uint32_t*)TaprootHash, hash160);

					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x04;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], TaprootHash, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}


				}

				if (ethereum) {
					keyLenSkip = 65 * pkField;
					unsigned char keccak_hash[32];
					const char* toHash;
					unsigned char* PubKey = &pubKeys[pkField * 65];
					unsigned char* start = PubKey + 1;
					toHash = reinterpret_cast<const char*>(start);
					keccak(toHash, 64, keccak_hash, 32);
					uint32_t hash160[8];
					for (int h = 12, i = 0; i < 5; i++) {
						hash160[i] = (keccak_hash[h++]) | ((keccak_hash[h++] << 8) & 0x0000ff00) | ((keccak_hash[h++] << 16) & 0x00ff0000) | ((keccak_hash[h++] << 24) & 0xff000000);
					}
					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x06;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (xpoint) {
					keyLenSkip = 65 * pkField;
					unsigned char* PubKey = &pubKeys[pkField * 65];
					uint32_t xpoint[8];
					unsigned char* start = PubKey + 1;
					memcpy(xpoint, start, 32);
					if (checkHash(xpoint)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x05;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &xpoint, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}


				}
				if (xrp) {
					keyLenSkip = 65 * pkField;
					_GetHash160Comp(pubKeys, keyLenSkip, (uint8_t*)hash160);
					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x90;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
				if (sui)
				{
					keyLenSkip = 65 * pkField;
					unsigned char* PubKey = &pubKeys[pkField * 65];
					uint8_t buff[34];
					unsigned char* start = PubKey + 1;
					memcpy(&buff[2], start, 32);
					buff[0] = 0x01;
					buff[1] = 0x2 + (PubKey[64] & 1);
					uint8_t hash_addr[32];
					Blake2b_256(buff, 34, hash_addr);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x70;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (iota)
				{
					keyLenSkip = 65 * pkField;
					unsigned char* PubKey = &pubKeys[pkField * 65];
					uint8_t buff[34];
					unsigned char* start = PubKey + 1;
					memcpy(&buff[2], start, 32);
					buff[0] = 0x01;
					buff[1] = 0x2 + (PubKey[64] & 1);
					uint8_t hash_addr[32];
					Blake2b_256(buff, 34, hash_addr);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x50;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (aptos)
				{
					keyLenSkip = 65 * pkField;
					unsigned char* PubKey = &pubKeys[pkField * 65];
					uint8_t buff[35] = { 0x00 };
					unsigned char* start = PubKey + 1;
					memcpy(&buff[2], start, 32);
					buff[0] = 0x01;
					buff[1] = 0x2 + (PubKey[64] & 1);
					buff[34] = 0x02;
					uint8_t hash_addr[32];
					sha3_256((char*)buff, 35, hash_addr);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x22;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
				if (icp)
				{

					uint8_t principal[29] = { 0 };
					unsigned char* PubKey = &pubKeys[pkField * 65];
					icp_principal_from_secp256k1(&PubKey[0], &principal[0]);
					uint8_t hash_addr[32];
					icp_account_identifier(&principal[0], NULL, &hash_addr[0]);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x53;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
				if (fil)
				{
					uint8_t hash_addr[20];
					unsigned char* PubKey = &pubKeys[pkField * 65];
					Blake2b_160(&PubKey[0], 65, &hash_addr[0]);

					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x41;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 20);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					unsigned char keccak_hash[32];
					keccak((char*)&PubKey[1], 64, keccak_hash, 32);
					uint32_t hash160[8];
					for (int h = 12, i = 0; i < 5; i++) {
						hash160[i] = (keccak_hash[h++]) | ((keccak_hash[h++] << 8) & 0x0000ff00) | ((keccak_hash[h++] << 16) & 0x00ff0000) | ((keccak_hash[h++] << 24) & 0xff000000);
					}

					if (checkHash((uint32_t*)hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x42;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash160, 20);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (xtz)
				{
					uint8_t buff[33] = { 0x00 };
					unsigned char* PubKey = &pubKeys[pkField * 65];
					memcpy(&buff[1], PubKey + 1, 32);
					buff[0] = 0x2 + (PubKey[64] & 1);

					uint8_t hash_addr[20];
					Blake2b_160(&buff[0], 33, &hash_addr[0]);

					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x92;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &prvKeys[pkField * 32], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 20);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
			}
		}
		if (ed25519_dev)
		{
			ed25519_key_to_pub_batch(prvKeys, pubKeysED, privKeyIx);


			for (int pkField = 0; pkField < THREAD_STEPS_BRAIN && pkField < privKeyIx; pkField++)
			{

				int pk4 = pkField * 32;
				unsigned char* publ = &pubKeysED[pk4];
				unsigned char* pkey = &prvKeys[pk4];

				if (solana)
				{
					if (checkHash((uint32_t*)publ)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x60;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &publ[0], 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (ton)
				{
					uint8_t v3r1[32];
					pubkey_to_hash_ton(publ, "v3r1", v3r1);
					if (checkHash((uint32_t*)v3r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x85;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v3r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v3r2[32];
					pubkey_to_hash_ton(publ, "v3r2", v3r2);

					if (checkHash((uint32_t*)v3r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x86;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v3r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v4r2[32];
					pubkey_to_hash_ton(publ, "v4r2", v4r2);
					if (checkHash((uint32_t*)v4r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x88;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v4r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					uint8_t v5r1[32];
					pubkey_to_hash_ton(publ, "v5r1", v5r1);
					if (checkHash((uint32_t*)v5r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x89;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v5r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					uint8_t hv3[32];
					pubkey_to_hash_ton(publ, "hv3", hv3);

					if (checkHash((uint32_t*)hv3)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x8c;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hv3, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (ton_all)
				{
					uint8_t v1r1[32];
					pubkey_to_hash_ton(publ, "v1r1", v1r1);
					if (checkHash((uint32_t*)v1r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x80;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v1r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v1r2[32];
					pubkey_to_hash_ton(publ, "v1r2", v1r2);
					if (checkHash((uint32_t*)v1r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x81;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v1r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v1r3[32];
					pubkey_to_hash_ton(publ, "v1r3", v1r3);
					if (checkHash((uint32_t*)v1r3)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x82;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v1r3, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v2r1[32];
					pubkey_to_hash_ton(publ, "v2r1", v2r1);
					if (checkHash((uint32_t*)v2r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x83;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v2r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v2r2[32];
					pubkey_to_hash_ton(publ, "v2r2", v2r2);
					if (checkHash((uint32_t*)v2r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x84;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v2r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v3r1[32];
					pubkey_to_hash_ton(publ, "v3r1", v3r1);
					if (checkHash((uint32_t*)v3r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x85;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v3r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v3r2[32];
					pubkey_to_hash_ton(publ, "v3r2", v3r2);
					if (checkHash((uint32_t*)v3r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x86;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v3r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v4r1[32];
					pubkey_to_hash_ton(publ, "v4r1", v4r1);
					if (checkHash((uint32_t*)v4r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x87;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v4r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t v4r2[32];
					pubkey_to_hash_ton(publ, "v4r2", v4r2);
					if (checkHash((uint32_t*)v4r2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x88;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v4r2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					uint8_t v5r1[32];
					pubkey_to_hash_ton(publ, "v5r1", v5r1);
					if (checkHash((uint32_t*)v5r1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x89;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &v5r1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					uint8_t hv1[32];
					pubkey_to_hash_ton(publ, "hv1", hv1);
					if (checkHash((uint32_t*)hv1)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x8a;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hv1, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t hv2[32];
					pubkey_to_hash_ton(publ, "hv2", hv2);
					if (checkHash((uint32_t*)hv2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x8b;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hv2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

					uint8_t hv3[32];
					pubkey_to_hash_ton(publ, "hv3", hv3);
					if (checkHash((uint32_t*)hv3)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x8c;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);

							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hv3, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (xrp) {
					int keyLenSkip = 0;
					_GetHash160ED(publ, keyLenSkip, (uint8_t*)hash160);
					if (checkHash(hash160)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x91;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {
							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash160, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}


				}
				if (aptos)
				{
					uint8_t buff[33] = { 0x00 };
					uint8_t buff2[34] = { 0x00 };
					memcpy(buff, publ, 32);
					memcpy(&buff2[1], publ, 32);
					buff2[33] = 0x02;
					uint8_t hash_addr[32];
					uint8_t hash_addr2[32];
					sha3_256((char*)buff, 33, hash_addr);
					sha3_256((char*)buff2, 34, hash_addr2);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x20;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					if (checkHash((uint32_t*)hash_addr2)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x21;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr2, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
				if (sui)
				{
					uint8_t buff[33] = { 0x00 };
					memcpy(&buff[1], publ, 32);
					uint8_t hash_addr[32];
					Blake2b_256(buff, 33, hash_addr);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x71;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (iota)
				{
					uint8_t hash_addr[32];
					Blake2b_256(publ, 32, hash_addr);
					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x51;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (dot)
				{
					unsigned char public_key_sr[32];
					sr25519_keypair keypair = { 0 };
					sr25519_keypair_from_seed(keypair, pkey);
					memcpy(public_key_sr, keypair + 64, 32);
					if (checkHash((uint32_t*)public_key_sr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x31;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &public_key_sr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
					if (checkHash((uint32_t*)publ)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x30;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &publ[0], 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}

				}
				if (icp)
				{

					uint8_t principal[29] = { 0 };
					icp_principal_from_ed25519(&publ[0], &principal[0]);
					uint8_t hash_addr[32];
					icp_account_identifier(&principal[0], NULL, &hash_addr[0]);

					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x52;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 32);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}
				if (xtz)
				{
					uint8_t hash_addr[20];
					Blake2b_160(&publ[0], 32, &hash_addr[0]);

					if (checkHash((uint32_t*)hash_addr)) {
						buffResult[0] = true;
						isResult[0] = true;
						uint8_t type = 0x93;
						unsigned long long int idx = atomicAdd(&d_resultsCount[0], 1);
						if (idx < MAX_FOUNDS_DEV) {

							memcpy(d_foundStrings[idx], &toHashBuffer[pkField], len);
							memcpy(d_len[idx], &len, 4); memcpy(d_iter[idx], &ITERATION, 4);
							memcpy(d_foundPrvKeys[idx], &pkey[0], 32);
							memcpy(d_foundHash160[idx], &hash_addr, 20);
							d_type[idx] = type; d_round[idx] = current_round;
						}
					}
				}

			}
		}

		if (i < 2 * round)
		{
			int zeroz[32] = { 0 };
			int wrapped = bump_all_keys(&prvKeys[0], privKeyIx, 1, true, &zeroz[0]);
			if (secp256_dev)
			{
				if (wrapped)
				{

					for (int n = 0; n < privKeyIx; n++)
					{
						if (zeroz[n] != 0)
						{
							memcpy(pubKeys + (zeroz[n] - 1) * 65, SECP_G65, 65);
						}

					}

				}
				pub_add_basepoint_batch_from_prec(&pubKeys[0], privKeyIx, +1, precPtr);
			}
		}

	}
}
