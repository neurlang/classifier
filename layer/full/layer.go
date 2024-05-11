package full

type FullLayer struct {
	size int
	bits byte
}

type Full struct {
	vec []bool
	bits byte
}

// New creates a new full layer with size and bits
func New(size int, bits byte) (o *FullLayer) {
	o = new(FullLayer)
	o.size = size
	o.bits = bits
	return
}

// Lay turns full layer into a combiner
func (i *FullLayer) Lay() (o *Full) {
	o = new(Full)
	o.vec = make([]bool, i.size)
	o.bits = i.bits
	return
}
