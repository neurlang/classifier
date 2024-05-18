// Package conv2d implements a 2D bit-convolution layer and combiner
package conv2d

import "fmt"
import "github.com/neurlang/classifier/layer"

type Conv2DLayer struct {
	width, height, subwidth, subheight, repeat int
	shift int
}

type Conv2D struct {
	vec []bool
	width, height, subwidth, subheight, repeat int
	shift int
}

// MustNew creates a new Conv2D layer with size, subsize and repeat
func MustNew(width, height, subwidth, subheight, repeat int) *Conv2DLayer {
	o, err := New2(width, height, subwidth, subheight, repeat, 0)
	if err != nil {
		panic(err.Error())
	}
	return o
}
// New creates a new Conv2D layer with size, subsize and repeat
func New(width, height, subwidth, subheight, repeat int) (o *Conv2DLayer, err error) {
	return New2(width, height, subwidth, subheight, repeat, 0)
}

// MustNew/ creates a new Conv2D layer with size, subsize and repeat
func MustNew2(width, height, subwidth, subheight, repeat int, shift int) *Conv2DLayer {
	o, err := New(width, height, subwidth, subheight, repeat)
	if err != nil {
		panic(err.Error())
	}
	return o
}
// New/ creates a new Conv2D layer with size, subsize and repeat
func New2(width, height, subwidth, subheight, repeat int, shift int) (o *Conv2DLayer, err error) {
	if width < subwidth {
		return nil, fmt.Errorf("New Conv2D: Width %d is lower than Subwidth %d", width, subwidth)
	}
	if height < subheight {
		return nil, fmt.Errorf("New Conv2D: Height %d is lower than Subheight %d", height, subheight)
	}
	o = new(Conv2DLayer)
	o.width = width
	o.height = height
	o.subwidth = subwidth
	o.subheight = subheight
	o.repeat = repeat
	o.shift = shift
	return
}

// Lay turns Conv2D layer into a combiner
func (i *Conv2DLayer) Lay() (layer.Combiner) {
	var o Conv2D
	o.vec = make([]bool, i.width * i.height * i.repeat)
	o.width = i.width
	o.height = i.height
	o.subwidth = i.subwidth
	o.subheight = i.subheight
	o.repeat = i.repeat
	return &o
}
