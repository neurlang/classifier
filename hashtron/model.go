package hashtron

type Hashtron struct {
	program [][2]uint32
	bits    byte
}

func (h Hashtron) Get(n int) (s uint32, max uint32) {
	return h.program[n][0], h.program[n][1]
}
func (h Hashtron) Len() int {
	return len(h.program)
}
func (h Hashtron) Bits() byte {
	return h.bits
}
