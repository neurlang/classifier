package crossattention

// Put inserts a boolean at position n.
func (f *CrossAttention) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.

func (f *CrossAttention) Feature(n int) (o uint32) {
	if f.qkv {
		iov := n % 3 // 0=Query, 1=Key, 2=Value
		headSize := f.dim * 3
		headStart := (n / headSize) * headSize
		currentPos := (n % headSize) / 3 // Position within head

		// Query positions - never masked
		if iov == 0 {
			for x := 0; x < f.dim; x += 3 {
				// Mask future positions for keys/values
				if f.use_masking && (x/3) > currentPos {
					continue
				}
				query := f.vec[headStart+x+0]
				key := f.vec[headStart+x+1]
				val := f.vec[headStart+x+2]
				if query && key && val {
					o++
				}
			}
			return
		}

		// Key/Value positions - apply masking
		for x := 0; x < f.dim; x += 3 {
			queryPos := headStart + x
			nPos := n
			// Mask future positions for keys/values
			if f.use_masking && (x/3) > currentPos {
				continue
			}

			if iov == 1 { // Key matching
				if f.vec[queryPos] && f.vec[nPos] == f.vec[queryPos+2] {
					o++
				}
			} else { // Value aggregation
				if f.vec[queryPos] && f.vec[queryPos+1] && f.vec[nPos] {
					o++
				}
			}
		}
		return
	}

	// --- Original KeyValue Mode ---
	iov := n & 1
	dim := f.dim
	beginhead := (n / dim) * dim
	currentPos := n % dim // Position within the current head

	for x := iov ^ 1; x < dim; x += 2 {
		if f.use_masking {
			// Mask future in KeyValue mode
			if (x / 2) > currentPos {
				continue
			}
		}
		if f.vec[beginhead+x] && f.vec[n] {
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
