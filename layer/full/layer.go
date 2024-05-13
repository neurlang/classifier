// Package full implements a fully connected layer and combiner
package full

import "github.com/neurlang/classifier/layer"

type FullLayer struct {
	size int
	bits byte
	maxbits byte
}

type Full struct {
	vec []bool
	bits byte
	maxbits byte
}

// MustNew creates a new full layer with size and bits
func MustNew(size int, bits, maxbits byte) *FullLayer {
	o, err := New(size, bits, maxbits)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(size int, bits, maxbits byte) (o *FullLayer, err error) {
	o = new(FullLayer)
	o.size = size
	o.bits = bits
	o.maxbits = maxbits
	return
}

// Lay turns full layer into a combiner
func (i *FullLayer) Lay() layer.Combiner {
	o := new(Full)
	o.vec = make([]bool, i.size)
	o.bits = i.bits
	o.maxbits = i.maxbits
	return o
}
