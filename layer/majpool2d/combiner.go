package majpool2d

// Put sets the n-th bool directly.
func (s *MajPool2D) Put(n int, v bool) {
	s.vec[n] = v
}

// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
func (s *MajPool2D) Disregard(n int) bool {
	orign := n
	submatrix := s.subwidth * s.subheight
	matrix := s.width * s.height * submatrix
	base := (n / matrix) * matrix
	n %= matrix
	n /= submatrix
	n *= submatrix
	var w0, w1 int
	for m := 0; m < submatrix; m++ {
		if orign == base+m+n {
			w0++
			w1--
			continue
		}
		if (s.vec)[base+m+n] {
			w0++
			w1++
		} else {
			w0--
			w1--
		}
	}
	cond1 := (w0 > s.bias) == (w1 > s.bias)
	return cond1
}

// Feature returns the m-th feature from the combiner.
func (s *MajPool2D) Feature(m int) (o uint32) {
	supermatrix := s.width * s.height
	submatrix := s.subwidth * s.subheight
	matrix := supermatrix * submatrix
	base := (m / matrix) * matrix
	var startx, starty int
	if s.fixed {
		m %= matrix
		m /= submatrix
		starty = ((m) / s.width) * s.capheight
		startx = ((m) % s.width) * s.capwidth
	} else {
		starty = (m) / s.width
		startx = (m) % s.width
	}
	for y := 0; y < s.capheight; y++ {
		for x := 0; x < s.capwidth; x++ {
			var xx, yy int
			xx = (x + startx) % s.width
			yy = (y + starty) % s.height
			var w int
			for my := 0; my < s.subheight; my++ {
			for mx := 0; mx < s.subwidth; mx++ {
				if (s.vec)[base+(submatrix*(s.width*yy+xx))+(s.subwidth*my+mx)] {
					w++
				} else {
					w--
				}
			}}
			o <<= 1
			if w > s.bias {
				o |= 1
			}
		}
	}
	return
}
