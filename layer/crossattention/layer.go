// Package crossattention implements a cross attetion connected layer and combiner
package crossattention

import "github.com/neurlang/classifier/layer"

type CrossAttentionLayer struct {
	dim int
	heads int
}

type CrossAttention struct {
	vec []bool
	dim int
	heads int
}

// MustNew creates a new full layer with size and bits
func MustNew(dim int, heads int) *CrossAttentionLayer {
	o, err := New(dim, heads)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(dim int, heads int) (o *CrossAttentionLayer, err error) {
	o = new(CrossAttentionLayer)
	o.dim = dim
	o.heads = heads
	return
}

// Lay turns full layer into a combiner
func (i *CrossAttentionLayer) Lay() layer.Combiner {
	o := new(CrossAttention)
	o.vec = make([]bool, i.dim * i.heads)
	o.dim = i.dim
	o.heads = i.heads
	return o
}
