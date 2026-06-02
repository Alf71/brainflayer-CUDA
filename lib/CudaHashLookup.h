#ifndef _HASH_LOOKUP_HOST_H
#define _HASH_LOOKUP_HOST_H
//#include <vector>
//#include <string>
typedef struct hash160 {
	uint32_t h[5];

	hash160(const uint32_t hash[5])
	{
		memcpy(h, hash, sizeof(uint32_t) * 5);
	}
}hash160;

class CudaHashLookup {

private:
	uint8_t *_bloomFilterPtr[100];

	cudaError_t setTargetBloomFilter(std::vector<std::string>& targets);

	void cleanup();

public:

	CudaHashLookup()
	{
		for (int a = 0; a < 100; a++) {
			_bloomFilterPtr[a] = NULL;
		}
	}

	~CudaHashLookup()
	{
		cleanup();
	}

	cudaError_t setTargets(std::vector<std::string>& targets);
};

#endif
