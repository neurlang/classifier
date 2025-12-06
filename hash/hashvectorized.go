//go:build !noasm && amd64

package hash

import "github.com/klauspost/cpuid/v2"

func init() {
	// Check if the CPU supports AVX512
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		HashVectorized = hashAVX512Vectorized
		HashVectorizedDistinct = hashAVX512VectorizedDistinct
		hashVectorizedParallelism = 16
	} else {
		HashVectorized = hashNotVectorized
		HashVectorizedDistinct = hashNotVectorizedDistinct
		hashVectorizedParallelism = 1
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

func hashAVX512VectorizedDistinct(out []uint32, n []uint32, s []uint32, max []uint32) {
	hashVectorizedDistinctAVX512(&out[0], &n[0], &s[0], &max[0], uint32(len(out)))
	// self-checking
	//for i := range out {
	//	var ok = Hash(n[i], s[i], max[i])
	//	if out[i] != ok {
	//		println("result is wrong", i, out[i], ok)
	//		out[i] = ok
	//	}
	//}
}

var lCPI0_0 = [16]uint32{1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31}

func hashVectorizedAVX512(out *uint32, n *uint32, s *uint32, max, length uint32)
func hashVectorizedDistinctAVX512(out *uint32, n *uint32, s *uint32, max *uint32, length uint32)
