// Package sochastic implements a sochastic connected layer and combiner
package sochastic

import "github.com/neurlang/classifier/layer"

type SochasticLayer struct {
	size    int
	maxbits byte
	seed    uint32
}

type Sochastic struct {
	vec     []bool
	maxbits byte
	seed    uint32
}

// MustNew creates a new full layer with size and bits
func MustNew(size int, maxbits byte, seed uint32) *SochasticLayer {
	o, err := New(size, maxbits, seed)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(size int, maxbits byte, seed uint32) (o *SochasticLayer, err error) {
	o = new(SochasticLayer)
	o.size = size
	o.maxbits = maxbits
	o.seed = seed
	return
}

// Lay turns full layer into a combiner
func (i *SochasticLayer) Lay() layer.Combiner {
	o := new(Sochastic)
	o.vec = make([]bool, i.size)
	o.maxbits = i.maxbits
	o.seed = i.seed
	return o
}
