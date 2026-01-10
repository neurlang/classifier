// Package Hashtron implements a hashtron (classifier)
package hashtron

// Hashtron represents individual hashtron (classifier) in memory
type Hashtron struct {
	program [][2]uint32
	bits    byte

	quaternary []byte
}

// Push pushes the hashing command to position 0
func (h *Hashtron) Push(data [2]uint32) {
	h.program = append([][2]uint32{data}, h.program...)
}

// Get gets the hashing command at position n
func (h Hashtron) Get(n int) (s uint32, max uint32) {
	return h.program[n][0], h.program[n][1]
}

// Len gets the number of hashing commands (size of hashtron program)
func (h Hashtron) Len() int {
	return len(h.program)
}

// LenQ gets the size of learned data (size of quaternary filter)
func (h Hashtron) Len() int {
	return len(h.quaternary)
}

// Bits determines the number of output bits returned by hashtron using Forward
func (h Hashtron) Bits() byte {
	return h.bits
}

// SetBits sets the number of output bits returned by hashtron using Forward
func (h *Hashtron) SetBits(bits byte) {
	h.bits = bits
}
