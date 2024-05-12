// Package Hashtron implements a hashtron (artifical neuron)
package hashtron

// Hashtron represents individual hashtron (artifical neuron) in memory
type Hashtron struct {
	program [][2]uint32
	bits    byte
}

// Get gets the hashing command at position n
func (h Hashtron) Get(n int) (s uint32, max uint32) {
	return h.program[n][0], h.program[n][1]
}

// Len gets the number of hashing commands (size of hashtron program)
func (h Hashtron) Len() int {
	return len(h.program)
}

// Bits determines the number of output bits returned by hashtron using Forward
func (h Hashtron) Bits() byte {
	return h.bits
}

// SetBits sets the number of output bits returned by hashtron using Forward
func (h Hashtron) SetBits(bits byte)  {
	h.bits = bits
}
