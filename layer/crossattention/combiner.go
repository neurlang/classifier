package crossattention

// Put inserts a boolean at position n.
func (f *CrossAttention) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *CrossAttention) Feature(n int) (o uint32) {
	if f.qkv {
		iov := n % 3
		dim := f.dim
		beginhead := (n / dim) * dim

		if iov == 2 {
			// Handle the value position
			for x := 0; x < dim; x += 3 {
				query := f.vec[beginhead + x]      // Get the query
				key := f.vec[beginhead + x + 1]    // Get the key
				me := f.vec[n]                     // Current value
				if query && key && me {
					o++
				}
			}
		} else {
			// Handle query and key positions
			for x := 0; x < dim; x += 3 {
				others := f.vec[beginhead + x + iov] // Get the query or key based on iov
				value := f.vec[beginhead + x + 2]    // Get the value
				me := f.vec[n]                       // Current query or key
				if others && me == value {
					o++
				}
			}
		}
		return o
	}

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
