// Package Inference implements the inference stage of the Neurlang classifier
package inference

import "github.com/neurlang/classifier/hash"

type Model interface {
	Get(n int) (s uint32, max uint32)
	Len() int
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
