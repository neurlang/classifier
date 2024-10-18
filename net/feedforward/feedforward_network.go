// Package feedforward implements a feedforward network type
package feedforward

import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/layer"
import "github.com/neurlang/classifier/datasets"
import "sync"

// Intermediate is an intermediate value used as both layer input and layer output in optimization
type Intermediate interface {

	// Feature extracts n-th feature from Intermediate
	Feature(n int) uint32

	// Disregard reports whether Intermediate doesn't regard n-th bit as affecting the output
	Disregard(n int) bool
}

// SingleValue is a single value returned by the final layer
type SingleValue uint32

// Feature extracts the feature from SingleValue
func (v SingleValue) Feature(n int) uint32 {
	return uint32(v)
}

// Disregard reports whether SingleValue doesn't regard n-th bit as affecting the output
func (v SingleValue) Disregard(n int) bool {
	return false
}

// FeedforwardNetworkInput is one individual input to the feedforward network
type FeedforwardNetworkInput interface {
	Feature(n int) uint32
}

// FeedforwardNetworkParityInput is one individual input to the feedforward network with parity
type FeedforwardNetworkParityInput interface {
	Feature(n int) uint32
	Parity() bool
}

// FeedforwardNetwork is the feedforward network
type FeedforwardNetwork struct {
	layers    [][]hashtron.Hashtron
	mapping   []byte
	combiners []layer.Layer
	premodulo []uint32
}

// Len returns the number of hashtrons which need to be trained inside the network.
func (f FeedforwardNetwork) Len() (o int) {
	for _, v := range f.layers {
		o += len(v)
	}
	return
}

// LenLayers returns the number of layers. Each Layer and Combiner counts as a layer here.
func (f FeedforwardNetwork) LenLayers() int {
	return len(f.layers)
}

// GetLayer gets the layer number of hashtron based on hashtron number. Returns -1 on failure.
func (f FeedforwardNetwork) GetLayer(n int) int {
	for i, v := range f.layers {
		if n < len(v) {
			return i
		}
		n -= len(v)
	}
	return -1
}

// GetPosition gets the position of hashtron within layer based on the overall
// hashtron number. Returns -1 on failure.
func (f FeedforwardNetwork) GetPosition(n int) int {
	for _, v := range f.layers {
		if n < len(v) {
			return n
		}
		n -= len(v)
	}
	return -1
}

// GetLayerPosition gets the position of hashtron within layer based on the
// overall hashtron number and overall layer. Returns -1 on failure.
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

// GetHashtron gets n-th hashtron pointer in the network. You can currently write
// a new hashtron into the pointer, but you might not be able to in the future.
func (f FeedforwardNetwork) GetHashtron(n int) *hashtron.Hashtron {
	for _, v := range f.layers {
		if n < len(v) {
			return &v[n]
		}
		n -= len(v)
	}
	return nil
}

// NewLayer adds a hashtron layer to the end of network with n hashtrons, each recognizing bits bits.
func (f *FeedforwardNetwork) NewLayer(n int, bits byte) {
	f.NewLayerP(n, bits, 0)
}

// NewLayerP adds a hashtron layer to the end of network with n hashtrons, each recognizing bits bits, and input feature pre-modulo.
func (f *FeedforwardNetwork) NewLayerP(n int, bits byte, premodulo uint32) {
	var layer = make([]hashtron.Hashtron, n)
	for i := range layer {
		h, _ := hashtron.New(nil, bits)
		layer[i] = *h
	}
	f.layers = append(f.layers, layer)
	f.mapping = append(f.mapping, bits)
	f.combiners = append(f.combiners, nil)
	f.premodulo = append(f.premodulo, premodulo)
}

// SetLayersP sets an input feature pre-modulo to layers.
func (f *FeedforwardNetwork) SetLayersP(premodulo uint32) {
	for n := range f.premodulo {
		f.premodulo[n] = premodulo
	}
}

// NewCombiner adds a combiner layer to the end of network
func (f *FeedforwardNetwork) NewCombiner(layer layer.Layer) {
	f.layers = append(f.layers, nil)
	f.mapping = append(f.mapping, 0)
	f.combiners = append(f.combiners, layer)
	f.premodulo = append(f.premodulo, 0)
}

// IsMapLayerOf checks if hashtron n lies in the final layer of the network.
func (f FeedforwardNetwork) IsMapLayerOf(n int) bool {
	if f.GetLayer(n) == -1 {
		return false
	}
	return f.mapping[f.GetLayer(n)] > 0
}

// Infer3 infers the network output based on input, after being trained by using Tally3
func (f FeedforwardNetwork) Infer3(input FeedforwardNetworkParityInput) (ouput FeedforwardNetworkInput) {
	in := f.Infer(FeedforwardNetworkInput(input))
	if input.Parity() {
		return tally3io{
			par: input,
			out: in,
		}
	}
	return in
}

// Infer infers the network output based on input, after being trained by using Tally2 or Tally
func (f FeedforwardNetwork) Infer(in FeedforwardNetworkInput) (ouput FeedforwardNetworkInput) {
	for l_prev := 0; l_prev < f.LenLayers(); l_prev += 2 {
		in, _ = f.Forward(in, l_prev, -1, 0)
	}
	return in
}

// Forward solves the intermediate value (net output after layer l based on that layer's input in) and the bit
// returned by worst hashtron is optionally negated (using neg == 1) and returned as computed.
func (f FeedforwardNetwork) Forward(in FeedforwardNetworkInput, l, worst, neg int) (inter Intermediate, computed bool) {
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		var combiner = f.combiners[l+1].Lay()
		wg := sync.WaitGroup{}
		for i := range f.layers[l] {
			wg.Add(1)
			go func(i int) {
				var feat = in.Feature(i)
				if f.premodulo[l] != 0 {
					feat = hash.Hash(feat, uint32(i), f.premodulo[l])
				}
				var bit = f.layers[l][i].Forward(feat, (i == worst) && (neg == 1))
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
	if len(f.mapping) > l && f.mapping[l] > 0 {
		if f.premodulo[l] != 0 {
			in = SingleValue(hash.Hash(in.Feature(0), 0, f.premodulo[l]))
		}
		var val = f.layers[l][0].Forward(in.Feature(0), (0 == worst) && (neg == 1))
		return SingleValue(val), false
	} else {
		if f.premodulo[l] != 0 {
			in = SingleValue(hash.Hash(in.Feature(0), 0, f.premodulo[l]))
		}
		var bit = f.layers[l][0].Forward(in.Feature(0), (0 == worst) && (neg == 1))
		return SingleValue(bit & 1), (bit & 1) != 0
	}
	return nil, false
}

type tally3io struct {
	par FeedforwardNetworkParityInput
	out FeedforwardNetworkInput
}

func (io tally3io) Feature(n int) uint32 {
	if io.par.Parity() {
		return io.out.Feature(n) ^ 1
	}
	return io.out.Feature(n)
}

// Tally3 tallies the network like Tally2, except it can also balance the dataset using input parity bit.
// Loss is 0 if the output is correct, below or equal to maxloss otherwise.
func (f *FeedforwardNetwork) Tally3(in FeedforwardNetworkParityInput, output FeedforwardNetworkInput,
	worst int, tally *datasets.Tally, loss func(i FeedforwardNetworkInput) uint32) {

	var newOut = tally3io{
		par: in,
		out: output,
	}

	output = newOut

	f.Tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
		return loss(i) < loss(j)
	})
}

// Tally2 tallies the network like Tally, except it can also optimize n-way classifiers. Loss is 0 if the
// output is correct, below or equal to maxloss otherwise.
func (f *FeedforwardNetwork) Tally2(in, output FeedforwardNetworkInput, worst int, tally *datasets.Tally,
	loss func(i FeedforwardNetworkInput) uint32) {
	l := f.GetLayer(worst)
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		f.Tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
			return loss(i) < loss(j)
		})
		return
	}
	if len(f.mapping) > l && f.mapping[l] > 0 {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint16(in.Feature(0))
		if f.premodulo[l] != 0 {
			ifeature = uint16(hash.Hash(uint32(ifeature), 0, f.premodulo[l]))
		}
		//out, _ := f.Forward(in, l, -1, 0)

		tally.AddToMapAll(ifeature, uint64(output.Feature(0)), func(n uint32) uint32 {
			return loss(SingleValue(n))
		}, uint32(1)<<f.mapping[l])
	} else {
		f.Tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
			return loss(i) < loss(j)
		})
		return
	}
}

// Tally tallies the network on input/output pair with respect to to-be-trained worst hashtron.
// The tally is stored into thread safe structure Tally. Two ouputs i, j can be compared to be less
// worse using the function less (returning true if output i is less worse than output j).
func (f *FeedforwardNetwork) Tally(in, output FeedforwardNetworkInput, worst int, tally *datasets.Tally, less func(i, j FeedforwardNetworkInput) bool) {
	l := f.GetLayer(worst)
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		var predicted [2]FeedforwardNetworkInput
		var compute [2]int8

		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifw := in.Feature(f.GetPosition(worst))
		if f.premodulo[l] != 0 {
			ifw = hash.Hash(ifw, uint32(f.GetPosition(worst)), f.premodulo[l])
		}
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
				inter, _ = f.Forward(inter, l_post, -1, 0)
			}
			predicted[neg] = inter
		}

		if !less(predicted[0], output) && !less(predicted[1], output) &&
			!less(output, predicted[0]) && !less(output, predicted[1]) {
			// we are correct anyway
			return
		}
		for neg := 0; neg < 2; neg++ {
			if !less(predicted[neg], output) && !less(output, predicted[neg]) {

				tally.AddToCorrect(ifw, compute[neg], neg == 1)
				// shift to correct output

				return
			}
		}
		// Further refined part
		if less(output, predicted[0]) != less(output, predicted[1]) {
			// Output is between the two predictions
			if less(output, predicted[0]) {
				// shift away from wrong
				tally.AddToImprove(ifw, -compute[0])
			} else {
				// shift away from wrong
				tally.AddToImprove(ifw, -compute[1])
			}
		} else {
			// Output is below or above
			if less(output, predicted[1]) {
				// shift towards better
				tally.AddToImprove(ifw, compute[0])
			} else {
				// shift towards better
				tally.AddToImprove(ifw, compute[1])
			}
		}
	}
	if len(f.mapping) > l && f.mapping[l] > 0 {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint16(in.Feature(0))
		if f.premodulo[l] != 0 {
			ifeature = uint16(hash.Hash(uint32(ifeature), 0, f.premodulo[l]))
		}
		tally.AddToMapping(ifeature, uint64(output.Feature(0)))
	} else {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint32(in.Feature(0))
		if f.premodulo[l] != 0 {
			ifeature = hash.Hash(uint32(ifeature), 0, f.premodulo[l])
		}
		tally.AddToCorrect(ifeature, 2*int8(output.Feature(0)&1)-1, true)
	}
}
