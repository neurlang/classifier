// Package majpool2d implements a 2D majority pooling layer and combiner
package majpool2d

import "github.com/neurlang/classifier/layer"

type MajPool2DLayer struct {
	width, height, subwidth, subheight, capwidth, capheight, repeat, bias int
}

type MajPool2D struct {
	vec                                                                   []bool
	width, height, subwidth, subheight, capwidth, capheight, repeat, bias int
}

// MustNew creates a new MajPool2D layer with size, subsize and repeat
func MustNew(width, height, subwidth, subheight, capwidth, capheight, repeat int) *MajPool2DLayer {
	o, err := New2(width, height, subwidth, subheight, capwidth, capheight, repeat, 0)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// New creates a new MajPool2D layer with size, subsize and repeat
func New(width, height, subwidth, subheight, capwidth, capheight, repeat, bias int) (*MajPool2DLayer, error) {
	return New2(width, height, subwidth, subheight, capwidth, capheight, repeat, 0)
}

// New2 creates a new MajPool2D layer with size, subsize and repeat
func New2(width, height, subwidth, subheight, capwidth, capheight, repeat, bias int) (o *MajPool2DLayer, err error) {
	o = new(MajPool2DLayer)
	o.width = width
	o.height = height
	o.subwidth = subwidth
	o.subheight = subheight
	o.capwidth = capwidth
	o.capheight = capheight
	o.repeat = repeat
	o.bias = bias
	return
}

// MustNew2 creates a new MajPool2D layer with size, subsize and repeat
func MustNew2(width, height, subwidth, subheight, capwidth, capheight, repeat, bias int) *MajPool2DLayer {
	o, err := New2(width, height, subwidth, subheight, capwidth, capheight, repeat, bias)
	if err != nil {
		panic(err.Error())
	}
	return o
}

// Lay turns MajPool2D layer into a combiner
func (i *MajPool2DLayer) Lay() layer.Combiner {
	var o MajPool2D
	o.vec = make([]bool, i.width*i.height*i.subwidth*i.subheight*i.repeat)
	o.width = i.width
	o.height = i.height
	o.subwidth = i.subwidth
	o.subheight = i.subheight
	o.capwidth = i.capwidth
	o.capheight = i.capheight
	o.repeat = i.repeat
	o.bias = i.bias
	return &o
}
