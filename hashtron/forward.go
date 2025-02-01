package hashtron

import "github.com/neurlang/classifier/hash"

// Forward calculates the hashtron output on single feaure sample (32 bit uint)
func (h Hashtron) Forward(sample uint32, negate bool) (out uint64) {
	for j := byte(0); j < h.Bits(); j++ {
		var input = uint32(sample) | (uint32(j) << 16)
		for i := 0; i < h.Len(); i++ {
			var s, max = h.Get(i)
			input = hash.Hash(input, s, max)
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
