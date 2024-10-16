package hash

import "github.com/klauspost/cpuid/v2"

func init() {
	// Check if the CPU supports AVX512
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		HashVectorized = hashAVX512Vectorized
	} else {
		HashVectorized = hashNotVectorized
	}
}

// HashVectorized implement many Neurlang hashes in parallel, using something like AVX-512 or similar
var HashVectorized func(out []uint32, n []uint32, s []uint32, max uint32)

func hashNotVectorized(out []uint32, n []uint32, s []uint32, max uint32) {
	for i := range out {
		out[i] = Hash(n[i], s[i], max)
	}
}
func hashAVX512Vectorized(out []uint32, n []uint32, s []uint32, max uint32) {
	hashVectorizedAVX512(&out[0], &n[0], &s[0], max, uint32(len(out)))
	// self-checking
	//for i := range out {
	//	var ok = Hash(n[i], s[i], max)
	//	if out[i] != ok {
	//		println("result is wrong", i, out[i], ok)
	//		out[i] = ok
	//	}
	//}
}

func hashVectorizedAVX512(out *uint32, n *uint32, s *uint32, max, length uint32)
