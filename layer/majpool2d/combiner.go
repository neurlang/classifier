package majpool2d

// Put sets the n-th bool directly and rotates the matrices if necessary.
func (s *MajPool2D) Put(n int, v bool) {
	s.vec[n] = v
}


// Disregard tells whether putting value false at position n would not affect
// any feature output (as opposed to putting value true at position n).
func (s *MajPool2D) Disregard(n int) bool {
	orign := n
	matrix := s.width * s.height * s.subwidth * s.subheight
	base := (n / matrix) * matrix
	n %= matrix
	n /= (s.subwidth * s.subheight)
	n *= (s.subwidth * s.subheight)
	var w int
	for m := 0; m < int(s.subheight)*int(s.subwidth); m++ {
		if orign == base + m + n {
			continue
		}
		if (s.vec)[base + m + n] {
			w++
		} else {
			w--
		}
	}
	return w != 0
}

// Feature returns the m-th feature from the combiner.
func (s *MajPool2D) Feature(m int) (o uint32) {
	matrix := s.width * s.height
	submatrix := s.subwidth * s.subheight
	base := (m / matrix) * submatrix
	m %= matrix
	y := m / s.width
	x := m % s.width
	subx := submatrix*x
	suby := submatrix * s.width * y
	for n := 0; n < submatrix; n++ {
		var w int
		for m := 0; m < int(s.subheight)*int(s.subwidth); m++ {
			if (s.vec)[base + subx + suby + n] {
				w++
			} else {
				w--
			}
		}
		if w > 0 {
			o |= 1 << n
		}
	}
	return 
}

