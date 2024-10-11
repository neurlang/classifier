package conv2d

// Put inserts a boolean at position n.
func (f *Conv2D) Put(n int, v bool) {
	f.vec[n] = v
}

// Feature returns the n-th feature from the combiner. Next layer reads
// its inputs using this method for hashtron n in the next layer.
func (f *Conv2D) Feature(n int) (o uint32) {

	block := (f.width - f.subwidth + 1) * (f.height - f.subheight + 1)
	nin := n % block
	nadd := (n / block) * block
	ny := nin / (f.height - f.subheight + 1)
	nx := nin % (f.width - f.subwidth + 1)

	var shift int
	for i := 1; i <= f.shift; i <<= 1 {
		shift++
	}

	for i := 0; i < f.subheight; i++ {
		for j := 0; j < f.subwidth; j++ {
			if f.shift != 0 {
				if (i*f.subwidth+j)%f.shift == 0 {
					o <<= shift
				}
			}
			if (f.vec)[nadd+(f.width*ny+nx)+(f.width*i+j)] {
				o++
			}
		}
	}
	return
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
// This excludes training samples early which would have no impact,
// because the next layer would see the same thing regardless of what we put.
func (f *Conv2D) Disregard(n int) bool {
	return false
}
