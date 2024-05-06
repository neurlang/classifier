// Package Inference implements the inference stage of the Neurlang classifier
package inference

import "github.com/neurlang/classifier/hash"

type Model interface {
	Get(n int) (s uint32, max uint32)
	Len() int
	Bits() byte
}

func BoolInfer(input uint32, m Model) bool {
	input = Uint32Infer(input, m)
	input &= 1
	return input != 0
}

func Uint16Infer2(input uint16, m0, m1 Model) (out uint16) {
	return Uint16Infer(Uint16Infer(input, m0), m1)
}
func Uint16Infer(command uint16, m Model) (out uint16) {
	if m.Len() == 0 {
		return
	}
	for j := byte(0); j < m.Bits(); j++ {
		var input = uint32(command) | (uint32(j) << 16)
		var ss, maxx = m.Get(0)
		input = hash.Hash(input, ss, maxx)
		for i := 1; i < m.Len(); i++ {
			var s, max = m.Get(i)
			maxx -= max
			input = hash.Hash(input, s, maxx)
		}
		input &= 1
		if input != 0 {
			out |= 1 << j
		}
	}
	return
}

func Uint32Infer(input uint32, m Model) uint32 {
	if m.Len() == 0 {
		return 0
	}
	var ss, maxx = m.Get(0)
	input = hash.Hash(input, ss, maxx)
	for i := 1; i < m.Len(); i++ {
		var s, max = m.Get(i)
		maxx -= max
		input = hash.Hash(input, s, maxx)
	}
	return input
}
