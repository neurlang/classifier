// Package sum implements a sum layer and combiner
package sum

// Put inserts a boolean at position n.
func (f *Sum) Put(n int, v bool) {
	f.vec[n].Store(v)
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *Sum) Feature(n int) (o uint32) {
	n *= int(f.dim)
	// Calculate sum using precomputed initial + j*step
	for j := uint(0); j < f.dim; j++ {
		if f.vec[(uint(n) + j * f.step) % uint(len(f.vec))].Load() {
			o++
		}
	}
	return
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
// This excludes training samples early which would have no impact,
// because the next layer would see the same thing regardless of what we put.
func (f *Sum) Disregard(n int) bool {
	return false
}
