package hashtron

import "errors"
import "math/rand"

// New creates a new hashtron (giving pseudo random output), or initialized using a provided program and bits
func New(program [][2]uint32, bits byte, future ...interface{}) (h *Hashtron, err error) {
	if future != nil && len(future) > 1 {
		return nil, errors.New("future arg not supported (new Hashtron)")
	}
	h = new(Hashtron)
	if bits == 0 {
		bits = 1
	}
	if program == nil {
		h.program = [][2]uint32{{rand.Uint32() >> 1, 2}}
	} else {
		h.program = program
	}
	if len(future) > 0 {
		if quaternary, ok := future[0].([]byte); ok {
			h.quaternary = quaternary
		}
	}
	h.bits = bits
	return
}
