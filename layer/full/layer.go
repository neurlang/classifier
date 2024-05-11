package full

type FullLayer struct {
	size int
	bits byte
}

type Full struct {
	vec []bool
	bits byte
}

// MustNew creates a new full layer with size and bits
func MustNew(size int, bits byte) *FullLayer {
	o, err := New(size, bits)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new full layer with size and bits
func New(size int, bits byte) (o *FullLayer, err error) {
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
