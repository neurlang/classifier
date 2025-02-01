// Package AVX implements the learning stage of the Neurlang classifier on AVX
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

// ScatterGatherVectorizedParallelism reports the recommended number of scatter gather to compute in parallel on this platform
// Can't return 0.
func ScatterGatherVectorizedParallelism() int {
	return scatterGatherVectorizedParallelism
}

// ScatterGatherVectorized implement many scatter gather in parallel, using something like AVX-512 or similar
var ScatterGatherVectorized func(outs []uint32, buf []uint32, size *int, wh uint32) bool

var scatterGatherVectorizedParallelism int


func scatterGatherAVX512Vectorized(outs []uint32, buf []uint32, size *int, wh uint32) bool

