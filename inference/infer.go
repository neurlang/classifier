// Package Hash implements the inference stage of the Neurlang classifier
package inference

import "github.com/neurlang/classifier/hash"

type Model interface {
	Get(n int) (s uint32, max uint32)
	Len() int
	Xor() uint32
}

func BoolInfer(input uint32, m Model) bool {
	for i := 0; i < m.Len(); i++ {
		var s, max = m.Get(i)
		input = hash.Hash(input, s, max)
	}
	input &= 1
	input ^= m.Xor()
	return input != 0
}
