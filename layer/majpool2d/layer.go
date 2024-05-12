// Package majpool2d implements a 2D majority pooling layer and combiner
package majpool2d

import "github.com/neurlang/classifier/layer"
import "fmt"

type MajPool2DLayer struct {
	width, height, subwidth, subheight, capwidth, capheight, repeat int
}

type MajPool2D struct {
	vec                                                             []bool
	width, height, subwidth, subheight, capwidth, capheight, repeat int
}

// New creates a new MajPool2D layer with size, subsize and repeat
func New(width, height, subwidth, subheight, capwidth, capheight, repeat int) (o *MajPool2DLayer, err error) {
	submatrix := subwidth * subheight
	if submatrix&1 == 0 {
		return nil, fmt.Errorf("even matrix")
	}
	o = new(MajPool2DLayer)
	o.width = width
	o.height = height
	o.subwidth = subwidth
	o.subheight = subheight
	o.capwidth = capwidth
	o.capheight = capheight
	o.repeat = repeat
	return
}

// MustNew creates a new MajPool2D layer with size, subsize and repeat
func MustNew(width, height, subwidth, subheight, capwidth, capheight, repeat int) *MajPool2DLayer {
	o, err := New(width, height, subwidth, subheight, capwidth, capheight, repeat)
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
	return &o
}
