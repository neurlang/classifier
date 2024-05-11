package feedforward

import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/layer"
import "github.com/neurlang/classifier/datasets"
import "sync"

type Intermediate interface {
	Feature(n int) uint32
	Disregard(n int) bool
}

// SingleValue

type SingleValue uint32

func (v SingleValue) Feature(n int) uint32 {
	return uint32(v)
}
func (v SingleValue) Disregard(n int) bool {
	return false
}

type FeedforwardNetworkInput interface {
	Feature(n int) uint32
}

type FeedforwardNetwork struct {
	layers    [][]hashtron.Hashtron
	mapping   []bool
	combiners []layer.Layer
}

func (f FeedforwardNetwork) Len() (o int) {
	for _, v := range f.layers {
		o += len(v)
	}
	return
}

func (f FeedforwardNetwork) LenLayers() int {
	return len(f.layers)
}

func (f FeedforwardNetwork) GetLayer(n int) int {
	for i, v := range f.layers {
		if n < len(v) {
			return i
		}
		n -= len(v)
	}
	return -1
}

func (f FeedforwardNetwork) GetPosition(n int) int {
	for _, v := range f.layers {
		if n < len(v) {
			return n
		}
		n -= len(v)
	}
	return -1
}

func (f FeedforwardNetwork) GetLayerPosition(l, n int) int {
	for i, v := range f.layers {
		if n < len(v) {
			if l == i {
				return n
			} else {
				return -1
			}
		}
		n -= len(v)
	}
	return -1
}

func (f FeedforwardNetwork) GetHashtron(n int) *hashtron.Hashtron {
	for _, v := range f.layers {
		if n < len(v) {
			return &v[n]
		}
		n -= len(v)
	}
	return nil
}

func (f *FeedforwardNetwork) NewLayer(n int, bits byte) {
	var layer = make([]hashtron.Hashtron, n)
	for i := range layer {
		h, _ := hashtron.New(nil, bits)
		layer[i] = *h
	}
	f.layers = append(f.layers, layer)
	f.mapping = append(f.mapping, bits > 1)
	f.combiners = append(f.combiners, nil)
}

func (f *FeedforwardNetwork) NewCombiner(layer layer.Layer) {
	f.layers = append(f.layers, nil)
	f.mapping = append(f.mapping, false)
	f.combiners = append(f.combiners, layer)
}

func (f FeedforwardNetwork) IsMapLayerOf(n int) bool {
	if f.GetLayer(n) == -1 {
		return false
	}
	return f.mapping[f.GetLayer(n)]
}
func (f FeedforwardNetwork) Infer(in FeedforwardNetworkInput) (ouput FeedforwardNetworkInput) {
	for l_prev := 0; l_prev < f.LenLayers(); l_prev += 2 {
		in, _ = f.Forward(in, l_prev, -1, 0)
	}
	return in
}

func (f FeedforwardNetwork) Forward(in FeedforwardNetworkInput, l, worst, neg int) (inter Intermediate, computed bool) {
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		var combiner = f.combiners[l+1].Lay()
		wg := sync.WaitGroup{}
		for i := range f.layers[l] {
			wg.Add(1)
			go func(i int) {
				var bit = f.layers[l][i].Forward(in.Feature(i), (i == worst) && (neg == 1))
				combiner.Put(i, bit&1 != 0)
				if i == worst {
					computed = bit&1 != 0
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
		return combiner, computed
	}
	if len(f.mapping) > l && f.mapping[l] {
		var val = f.layers[l][0].Forward(in.Feature(0), (0 == worst) && (neg == 1))
		return SingleValue(val), false
	} else {
		var bit = f.layers[l][0].Forward(in.Feature(0), (0 == worst) && (neg == 1))
		return SingleValue(bit & 1), (bit & 1) != 0
	}
	return nil, false
}

func (f *FeedforwardNetwork) Tally(in, output FeedforwardNetworkInput, worst int, tally datasets.Tally, less func(i, j FeedforwardNetworkInput) bool) {
	l := f.GetLayer(worst)
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		var predicted [2]FeedforwardNetworkInput
		var compute [2]int8

		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifw := in.Feature(f.GetPosition(worst))
		for neg := 0; neg < 2; neg++ {
			inter, computed := f.Forward(in, l, f.GetPosition(worst), neg)
			if computed {
				compute[neg] = 1
			} else {
				compute[neg] = -1
			}
			if neg == 0 {
				if inter.Disregard(f.GetLayerPosition(l, worst)) {
					return
				}
			}
			for l_post := l + 2; l_post < f.LenLayers(); l_post += 2 {
				in, _ = f.Forward(in, l_post, -1, 0)
			}
			predicted[neg] = in
		}
		if !less(predicted[0], output) && !less(predicted[1], output) &&
			!less(output, predicted[0]) && !less(output, predicted[1]) {
			// we are correct anyway
			return
		}
		for neg := 0; neg < 2; neg++ {
			if !less(predicted[neg], output) && !less(output, predicted[neg]) {

				tally.AddToCorrect(ifw, compute[neg])
				// shift to correct output

				return
			}
		}
		if less(predicted[0], predicted[1]) {
			// shift away to less wrong output
			tally.AddToImprove(ifw, compute[0])
		} else {
			// shift away to less wrong output
			tally.AddToImprove(ifw, compute[1])
		}
	}
	if len(f.mapping) > l && f.mapping[l] {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint16(in.Feature(0))

		tally.AddToMapping(ifeature, uint64(output.Feature(0)))
	} else {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint32(in.Feature(0))

		tally.AddToCorrect(ifeature, 1)
	}
}
