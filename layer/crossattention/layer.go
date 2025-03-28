// Package crossattention implements a cross attetion connected layer and combiner
package crossattention

import "github.com/neurlang/classifier/layer"

type CrossAttentionLayer struct {
	qkv         bool
	dim         int
	heads       int
	use_masking bool
}

type CrossAttention struct {
	vec         []bool
	qkv         bool
	dim         int
	heads       int
	use_masking bool
}

// MustNew4 creates a new masked qkv full layer with size and bits
func MustNew4(dim int, heads int) *CrossAttentionLayer {
	o, err := New4(dim, heads)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New4 creates a new masked qkv full layer with size and bits
func New4(dim int, heads int) (o *CrossAttentionLayer, err error) {
	o = new(CrossAttentionLayer)
	o.dim = dim
	o.heads = heads
	o.qkv = true
	o.use_masking = true
	return
}

// MustNew3 creates a new qkv full layer with size and bits
func MustNew3(dim int, heads int) *CrossAttentionLayer {
	o, err := New3(dim, heads)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New3 creates a new qkv full layer with size and bits
func New3(dim int, heads int) (o *CrossAttentionLayer, err error) {
	o = new(CrossAttentionLayer)
	o.dim = dim
	o.heads = heads
	o.qkv = true
	return
}

// MustNew2 creates a new masked full layer with size and bits
func MustNew2(dim int, heads int) *CrossAttentionLayer {
	o, err := New2(dim, heads)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New2 creates a new masked full layer with size and bits
func New2(dim int, heads int) (o *CrossAttentionLayer, err error) {
	o = new(CrossAttentionLayer)
	o.dim = dim
	o.heads = heads
	o.use_masking = true
	return
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
	o.vec = make([]bool, i.dim*i.heads)
	o.dim = i.dim
	o.heads = i.heads
	o.qkv = i.qkv
	o.use_masking = i.use_masking
	return o
}
