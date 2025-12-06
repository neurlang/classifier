// Package sum implements a sum layer and combiner
package sum

import "github.com/neurlang/classifier/layer"
import "sync/atomic"

type SumLayer struct {
	size uint
	step uint
	dim  uint
}

type Sum struct {
	vec  []atomic.Bool
	step uint
	dim  uint
}

// MustNew creates a new full layer with size and bits
func MustNew(dims []uint, dim uint) *SumLayer {
	o, err := New(dims, dim)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(dims []uint, dim uint) (o *SumLayer, err error) {
	var size = uint(1)
	for _, dim := range dims {
		size *= dim
	}
	// Precompute strides and validate inputs
	stride := uint(1)
	strides := make([]uint, len(dims))
	for i := range dims {
		strides[i] = stride
		stride *= dims[i]
	}
	step := strides[dim]

	o = new(SumLayer)
	o.size = size
	o.step = step
	o.dim = dims[dim]
	return
}

// Lay turns full layer into a combiner
func (i *SumLayer) Lay() layer.Combiner {
	o := new(Sum)
	o.vec = make([]atomic.Bool, i.size)
	o.step = i.step
	o.dim = i.dim
	return o
}
