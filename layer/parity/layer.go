// Package parity implements a parity layer and combiner
package parity

import "github.com/neurlang/classifier/layer"

type ParityLayer struct {
	size int
}

type Parity struct {
	vec []bool
}

// MustNew creates a new full layer with size and bits
func MustNew(size int) *ParityLayer {
	o, err := New(size)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(size int) (o *ParityLayer, err error) {
	o = new(ParityLayer)
	o.size = size
	return
}

// Lay turns full layer into a combiner
func (i *ParityLayer) Lay() layer.Combiner {
	o := new(Parity)
	o.vec = make([]bool, i.size)
	return o
}
