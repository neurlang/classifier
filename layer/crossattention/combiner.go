package crossattention

// Put inserts a boolean at position n.
func (f *CrossAttention) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *CrossAttention) Feature(n int) (o uint32) {
/*
	iov := (n % 3)
	if iov == 2 {
		return
	}

	dim := f.dim
	beginhead := (n / dim) * dim

	for x := iov ^ 1; x < dim; x += 3 {
		others := f.vec[beginhead + x]
		value := f.vec[beginhead + x + (1 << iov)]
		me := f.vec[n]
		if others && me == value {
			o++
		}
	}
*/
	iov := n & 1
	dim := f.dim
	beginhead := (n / dim) * dim
	for x := iov ^ 1; x < dim; x += 2 {
		others := f.vec[beginhead + x]
		me := f.vec[n]
		if others && me {
			o++
		}
	}
	return
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
// This excludes training samples early which would have no impact,
// because the next layer would see the same thing regardless of what we put.
func (f *CrossAttention) Disregard(n int) bool {
	return false
}
