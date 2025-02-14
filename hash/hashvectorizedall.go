package hash

// HashVectorized implement many Neurlang hashes in parallel, using something like AVX-512 or similar
var HashVectorized func(out []uint32, n []uint32, s []uint32, max uint32) = hashNotVectorized
var HashVectorizedDistinct func(out []uint32, n []uint32, s []uint32, max []uint32) = hashNotVectorizedDistinct

var hashVectorizedParallelism int = 1

// HashVectorizedParallelism reports the recommended number of hashes to compute in parallel on this platform
// Can't return 0.
func HashVectorizedParallelism() int {
	return hashVectorizedParallelism
}

func hashNotVectorized(out []uint32, n []uint32, s []uint32, max uint32) {
	for i := range out {
		out[i] = Hash(n[i], s[i], max)
	}
}
func hashNotVectorizedDistinct(out []uint32, n []uint32, s []uint32, max []uint32) {
	for i := range out {
		out[i] = Hash(n[i], s[i], max[i])
	}
}
