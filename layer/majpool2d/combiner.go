package majpool2d

// Put sets the n-th bool directly and rotates the matrices if necessary.
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
	var w int
	for m := 0; m < submatrix; m++ {
		if orign == base+m+n {
			continue
		}
		if (s.vec)[base+m+n] {
			w++
		} else {
			w--
		}
		if m == 0 && submatrix & 1 == 0 {
			w *= 2
		}
	}
	return w != 0
}

// Feature returns the m-th feature from the combiner.
func (s *MajPool2D) Feature(m int) (o uint32) {
	supermatrix := s.width * s.height
	submatrix := s.subwidth * s.subheight
	residualwidth := (s.width - s.capwidth + 1)
	matrix := supermatrix * submatrix
	base := (m / matrix) * matrix
	starty := m / residualwidth
	startx := m % residualwidth
	for y := 0; y < s.capheight; y++ {
		for x := 0; x < s.capwidth; x++ {
			var w int
			for m := 0; m < submatrix; m++ {
				if (s.vec)[base+submatrix*(s.width*(y+starty)+(x+startx))+m] {
					w++
				} else {
					w--
				}
				if m == 0 && submatrix & 1 == 0 {
					w *= 2
				}
			}
			o <<= 1
			if w > 0 {
				o |= 1
			}
		}
	}
	return
}
