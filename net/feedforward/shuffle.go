package feedforward

import "math/rand"

func (f FeedforwardNetwork) Shuffle(reverse bool) (o []int) {
	o = make([]int, f.Len(), f.Len())
	for i := range o {
		o[i] = i
	}
	var base = 0
	for i := range f.layers {
		rand.Shuffle(len(f.layers[i]), func(i, j int) { o[base+i], o[base+j] = o[base+j], o[base+i] })
		base += len(f.layers[i])
	}
	if reverse {
		for i := 0; 2*i < len(o); i++ {
			o[i], o[len(o)-i-1] = o[len(o)-i-1], o[i]
		}
	}
	return o
}
