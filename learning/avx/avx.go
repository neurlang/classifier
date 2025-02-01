//go:build !noasm && amd64
package avx

import "github.com/klauspost/cpuid/v2"

func init() {
	// Check if the CPU supports AVX512
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		ScatterGatherVectorized = scatterGatherAVX512Vectorized
		scatterGatherVectorizedParallelism = 16
	} else {
		ScatterGatherVectorized = scatterGatherNotVectorized
		scatterGatherVectorizedParallelism = 1
	}
}


func scatterGatherAVX512Vectorized(outs []uint32, buf []uint32, size *int, wh uint32) bool

