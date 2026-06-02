#ifndef __VRF_H__
#define __VRF_H__



__device__ Sr25519SignatureResult vrf_sign(sr25519_vrf_io inout, sr25519_vrf_proof proof, sr25519_vrf_proof_batchable proof_batchable, const sr25519_keypair keypair, const merlin_transcript *t);
__device__ Sr25519SignatureResult shorten_vrf(sr25519_vrf_proof proof, const sr25519_vrf_proof_batchable proof_batchable, const sr25519_public_key public_key, const merlin_transcript *t, const sr25519_vrf_output preout);
__device__ Sr25519SignatureResult vrf_verify(sr25519_vrf_io inout, sr25519_vrf_proof_batchable proof_batchable, const sr25519_public_key public_key, const merlin_transcript *t, const sr25519_vrf_output preout, const sr25519_vrf_proof proof);
__device__ void io_make_bytes(sr25519_vrf_raw_output raw_output, const sr25519_vrf_io inout, const uint8_t *context, const size_t context_length);




#endif
