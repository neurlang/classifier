package sochastic

import "github.com/neurlang/classifier/hash"

// Put inserts a boolean at position n.
func (f *Sochastic) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *Sochastic) Feature(n int) (o uint32) {
	for pos := 0; pos < int(f.maxbits); pos++ {
		o <<= 1
		if f.vec[hash.Hash(f.seed ^ uint32(n), uint32(pos), uint32(len(f.vec)))] {
			o |= 1
		}
	}
	return
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
// This excludes training samples early which would have no impact,
// because the next layer would see the same thing regardless of what we put.
func (f *Sochastic) Disregard(n int) bool {
	return false
}
