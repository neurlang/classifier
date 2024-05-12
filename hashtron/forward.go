package hashtron

import "github.com/neurlang/classifier/hash"

// Forward calculates the hashtron output on single feaure sample (32 bit uint)
func (h Hashtron) Forward(sample uint32, negate bool) (out uint64) {
	if h.Len() == 0 {
		return
	}
	for j := byte(0); j < h.Bits(); j++ {
		var input = uint32(sample) | (uint32(j) << 16)
		var ss, maxx = h.Get(0)
		input = hash.Hash(input, ss, maxx)
		for i := 1; i < h.Len(); i++ {
			var s, max = h.Get(i)
			maxx -= max
			input = hash.Hash(input, s, maxx)
		}
		input &= 1
		if negate {
			input ^= 1
		}
		if input != 0 {
			out |= 1 << j
		}
	}
	return
}
