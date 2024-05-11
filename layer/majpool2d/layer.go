package majpool2d

import "github.com/neurlang/classifier/layer"

type MajPool2DLayer struct {
	width, height, subwidth, subheight, repeat int
}

type MajPool2D struct {
	vec []bool
	width, height, subwidth, subheight, repeat int
}

// New creates a new MajPool2D layer with size, subsize and repeat
func New(width, height, subwidth, subheight, repeat int) (o *MajPool2DLayer, err error) {
	return MustNew(width, height, subwidth, subheight, repeat), nil
}

// MustNew creates a new MajPool2D layer with size, subsize and repeat
func MustNew(width, height, subwidth, subheight, repeat int) (o *MajPool2DLayer) {
	o = new(MajPool2DLayer)
	o.width = width
	o.height = height
	o.subwidth = subwidth
	o.subheight = subheight
	o.repeat = repeat
	return
}

// Lay turns MajPool2D layer into a combiner
func (i *MajPool2DLayer) Lay() (layer.Combiner) {
	var o MajPool2D
	o.vec = make([]bool, i.width * i.height * i.subwidth * i.subheight * i.repeat)
	o.width = i.width
	o.height = i.height
	o.subwidth = i.subwidth
	o.subheight = i.subheight
	o.repeat = i.repeat
	return &o
}
