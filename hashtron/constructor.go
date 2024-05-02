package hashtron

import "errors"

func New(program [][2]uint32, bits byte, future ...interface{}) (h *Hashtron, err error) {
	if future != nil && len(future) > 0 {
		return nil, errors.New("future arg not supported (new Hashtron)")
	}
	h = new(Hashtron)

	h.program = program
	h.bits = bits
	return
}
