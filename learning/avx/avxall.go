// Package AVX implements the learning stage of the Neurlang classifier on AVX
package avx

// ScatterGatherVectorizedParallelism reports the recommended number of scatter gather to compute in parallel on this platform
// Can't return 0.
func ScatterGatherVectorizedParallelism() int {
	return scatterGatherVectorizedParallelism
}

// ScatterGatherVectorized implement many scatter gather in parallel, using something like AVX-512 or similar
var ScatterGatherVectorized func(outs []uint32, buf []uint32, size *int, wh uint32) bool = scatterGatherNotVectorized

var scatterGatherVectorizedParallelism int = 1


