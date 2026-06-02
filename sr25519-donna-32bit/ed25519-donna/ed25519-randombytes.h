__device__ void
ed25519_randombytes_unsafe (void *p, size_t len) {

		uint8_t* buf = (uint8_t*)p;
		for (int i = 0; i < len; i++)
		{
			buf[i] = 0x01;
		}
	

}

