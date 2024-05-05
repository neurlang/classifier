package hashtron

import "errors"
import "math/rand"

func New(program [][2]uint32, bits byte, future ...interface{}) (h *Hashtron, err error) {
	if future != nil && len(future) > 0 {
		return nil, errors.New("future arg not supported (new Hashtron)")
	}
	h = new(Hashtron)
	if bits == 0 {
		bits = 1
	}
	if program == nil {
		h.program = [][2]uint32{{rand.Uint32()>>1, 2}}
	} else {
		h.program = program
	}
	h.bits = bits
	return
}
