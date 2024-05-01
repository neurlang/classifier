// Package Inference implements the inference stage of the Neurlang classifier
package inference

import "github.com/neurlang/classifier/hash"

type Model interface {
	Get(n int) (s uint32, max uint32)
	Len() int
	Bits() byte
}

func BoolInfer(input uint32, m Model) bool {
	var maxx uint32
	for i := 0; i < m.Len(); i++ {
		var s, max = m.Get(i)
		if i == 0 {
			maxx += max
		} else {
			maxx -= max
		}
		input = hash.Hash(input, s, maxx)
	}
	input &= 1
	return input != 0
}

func Uint16Infer2(input uint16, m0, m1 Model) (out uint16) {
	return Uint16Infer(Uint16Infer(input, m0), m1)
}
func Uint16Infer(command uint16, m Model) (out uint16) {
	for j := byte(0); j < m.Bits(); j++ {
		var input = uint32(command) | uint32(1 << (j + 16))
		var maxx uint32
		for i := 0; i < m.Len(); i++ {
			var s, max = m.Get(i)
			if i == 0 {
				maxx += max
			} else {
				maxx -= max
			}
			input = hash.Hash(input, s, maxx)
		}
		input &= 1
		if input != 0 {
			out |= 1 << j
		}
	}
	return
}
