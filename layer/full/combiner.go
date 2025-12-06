package full

// Put inserts a boolean at position n.
func (f *Full) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *Full) Feature(n int) (o uint32) {
	n *= int(f.bits)
	if n+int(f.maxbits) > len(f.vec) {
		return 0
	}
	for pos := n; pos < n+int(f.maxbits) && pos < len(f.vec); pos++ {
		o <<= 1
		if f.vec[pos] {
			o |= 1
		}
	}
	return
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
// This excludes training samples early which would have no impact,
// because the next layer would see the same thing regardless of what we put.
func (f *Full) Disregard(n int) bool {
	return false
}
