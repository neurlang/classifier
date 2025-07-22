// Package feedforward implements a feedforward network type
package feedforward

import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/layer"
import "github.com/neurlang/classifier/datasets"

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

// FeedforwardNetworkParityInOutput is one individual sample to the feedforward network with parity and expected network output
type FeedforwardNetworkParityInOutput interface {
	Feature(n int) uint32
	Parity() uint16
	Output() uint16
}

// FeedforwardNetwork is the feedforward network
type FeedforwardNetwork struct {
	layers    [][]hashtron.Hashtron
	mapping   []byte
	combiners []layer.Layer
	premodulo []uint32
	preadd    []byte
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

// LenHashtronLayers returns the number of hashtron layers. Each Layer (not Combiner) counts as a layer here.
func (f FeedforwardNetwork) LenHashtronLayers() (l int) {
	for _, v := range f.layers {
		if len(v) > 0 {
			l++
		}
	}
	return
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

// GetFrontBase returns the base number of hashtron for front layer. Each Layer and LayerPA counts as a front layer here.
func (f FeedforwardNetwork) GetFrontBase(n int) (o int) {
	return f.getFrontBase(n, true)
}

// GetFrontBase returns the offset number for front layer feature. Each Layer and LayerPA counts as a front layer here.
func (f FeedforwardNetwork) GetFrontOffset(n int) (o int) {
	return f.getFrontBase(n, false)
}

func (f FeedforwardNetwork) getFrontBase(n int, is_need_hashtron bool) (o int) {
	for i, v := range f.layers {
		if n == i {
			return
		}

		if f.preadd[i] == preAddition && i > 0 {
			o += len(v)
		} else if preAddition != f.preadd[i] && i == 0 {
			o += len(v)
		} else if is_need_hashtron {
			o += len(v)
		}
	}
	return
}

// LenLayers returns the number of front layers. Each Layer and LayerPA counts as a front layer here.
func (f FeedforwardNetwork) LenFrontLayers() (o int) {
	o++
	for i, v := range f.preadd {
		if v == preAddition && i > 0 {
			o++
		}
	}
	return
}

// GetLayer gets the layer number based on front layer number. Returns -1 on failure.
func (f FeedforwardNetwork) GetFrontLayer(n int) int {
	if n == 0 {
		return 0
	}
	n--
	for i, v := range f.preadd {
		if v == preAddition && i > 0 && n <= 0 {
			return i
		}
		if v == preAddition {
			n--
		}
	}
	return -1
}

// Forget.
func (f *FeedforwardNetwork) Forget() {
	for _, v := range f.layers {
		for j := range v {
			h, _ := hashtron.New(nil, v[j].Bits())
			if h != nil {
				v[j] = *h
			} else {
				v[j] = hashtron.Hashtron{}
			}
		}
	}
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

// NewLayer adds a hashtron layer to the end of network with n hashtrons, each recognizing bits-bits.
func (f *FeedforwardNetwork) NewLayer(n int, bits byte) {
	f.NewLayerP(n, bits, 0)
}

const noAddition = 0
const preAddition = 1
const interAddition = 2

// NewLayerPA adds a hashtron layer to the end of network with n hashtrons, each recognizing bits-bits, and input feature pre-modulo and pre-addition.
func (f *FeedforwardNetwork) NewLayerPA(n int, bits byte, premodulo uint32) {
	f.newLayerPAI(n, bits, premodulo, preAddition)
}

// NewLayerPI adds a hashtron layer to the end of network with n hashtrons, each recognizing bits-bits, and input feature pre-modulo and inter-addition.
func (f *FeedforwardNetwork) NewLayerPI(n int, bits byte, premodulo uint32) {
	f.newLayerPAI(n, bits, premodulo, interAddition)
}

// NewLayerP adds a hashtron layer to the end of network with n hashtrons, each recognizing bits-bits, and input feature pre-modulo.
func (f *FeedforwardNetwork) NewLayerP(n int, bits byte, premodulo uint32) {
	f.newLayerPAI(n, bits, premodulo, noAddition)
}

// newLayerPAI adds a hashtron layer to the end of network with n hashtrons, each recognizing bits-bits, and input feature pre-modulo and pre-addition/inter-addition.
func (f *FeedforwardNetwork) newLayerPAI(n int, bits byte, premodulo uint32, preaddition byte) {
	var layer = make([]hashtron.Hashtron, n)
	for i := range layer {
		h, _ := hashtron.New(nil, bits)
		layer[i] = *h
	}
	if bits == 0 {
		bits = 1
	}
	f.layers = append(f.layers, layer)
	f.mapping = append(f.mapping, bits)
	f.combiners = append(f.combiners, nil)
	f.premodulo = append(f.premodulo, premodulo)
	f.preadd = append(f.preadd, preaddition)
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
	f.preadd = append(f.preadd, noAddition)
}

// IsMapLayerOf checks if hashtron n lies in the final layer of the network.
func (f FeedforwardNetwork) IsMapLayerOf(n int) bool {
	if f.GetLayer(n) == -1 {
		return false
	}
	return f.mapping[f.GetLayer(n)] > 0
}

// Infer3 infers the network output based on input, after being trained by using Tally3
func (f FeedforwardNetwork) infer3(input FeedforwardNetworkParityInput) (ouput FeedforwardNetworkInput) {
	in := f.infer(FeedforwardNetworkInput(input))
	if input.Parity() {
		return tally3io{
			par: input,
			out: in,
		}
	}
	return in
}

type inferPreaddBase struct {
	add  FeedforwardNetworkInput
	base int
	in   FeedforwardNetworkInput
}

func (i *inferPreaddBase) Disregard(n int) bool {
	if val, ok := (i.in).(Intermediate); ok {
		return val.Disregard(n)
	}
	return false
}

func (i *inferPreaddBase) Feature(n int) uint32 {
	return i.in.Feature(n) + i.add.Feature(n+i.base)
}

type infer2io struct {
	io FeedforwardNetworkParityInOutput
}

func (io infer2io) Feature(n int) uint32 {
	return io.io.Feature(n)
}

func (io infer2io) Parity() uint16 {
	return io.io.Parity()
}

func (io infer2io) Output() uint16 {
	return io.io.Output()
}

func (io infer2io) Disregard(int) bool {
	return false
}

// Infer2 infers the network output based on input, after being trained by using Tally4. This applies parity.
func (f FeedforwardNetwork) Infer2(input FeedforwardNetworkParityInOutput) (val uint16) {
	if input.Parity() == 0 {
		ret := f.infer(input)
		for j := byte(0); j < 16 && j < f.GetLastCells(); j++ {
			val |= uint16(ret.Feature(int(j))) << uint16(j)
		}
		return val
	}
	output := Intermediate(infer2io{io: input})
	preadd_input := output
	for l_prev := 0; l_prev < f.LenLayers(); l_prev += 2 {
		if f.preadd[l_prev] == preAddition {
			output = Intermediate(&inferPreaddBase{add: input, in: output, base: f.GetFrontOffset(l_prev)})
		}
		if f.preadd[l_prev] == interAddition {
			output = Intermediate(&inferPreaddBase{add: preadd_input, in: output, base: 0})
		}
		preadd_input = output
		output, _ = f.Forward(output, l_prev, -1, 0)
	}
	for j := byte(0); j < 16 && j < f.GetLastCells(); j++ {
		val |= uint16(output.Feature(int(j))) << uint16(j)
	}
	return (val ^ input.Parity())
}

// Infer infers the network output based on input, after being trained by using Tally2 or Tally
func (f FeedforwardNetwork) infer(in FeedforwardNetworkInput) (ouput FeedforwardNetworkInput) {
	ouput = in
	preadd_input := ouput
	for l_prev := 0; l_prev < f.LenLayers(); l_prev += 2 {
		if f.preadd[l_prev] == preAddition {
			ouput = Intermediate(&inferPreaddBase{add: in, in: ouput, base: f.GetFrontOffset(l_prev)})
		}
		if f.preadd[l_prev] == interAddition {
			ouput = Intermediate(&inferPreaddBase{add: preadd_input, in: ouput, base: 0})
		}
		preadd_input = ouput
		ouput, _ = f.Forward(ouput, l_prev, -1, 0)
	}
	return
}

// Forward solves the intermediate value (net output after layer l based on that layer's input in) and the bit
// returned by worst hashtron is optionally negated (using neg == 1) and returned as computed.
func (f FeedforwardNetwork) Forward(in FeedforwardNetworkInput, l int, worstneg ...int) (inter Intermediate, computed []bool) {
	computed = []bool{false, false}
	var worst, neg = -1, -1
	if len(worstneg) >= 2 {
		worst = worstneg[0]
		neg = worstneg[1]
	}
	var worst1, neg1 = -1, -1
	if len(worstneg) >= 4 {
		worst1, neg1 = worstneg[2], worstneg[3]
	}
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		var combiner = f.combiners[l+1].Lay()
		w0 := worst
		if neg != 1 {
			w0 = -1
		}
		w1 := worst1
		if neg1 != 1 {
			w1 = -1
		}
		var features = make([]uint32, len(f.layers[l]), len(f.layers[l]))
		for i := range features {
			features[i] = in.Feature(i)
			if f.premodulo[l] != 0 {
				features[i] = hash.Hash(features[i], uint32(i), f.premodulo[l])
			}
		}
		var bits = hashtron.HashtronSlice(f.layers[l]).Forward(features, w0, w1)
		for i, bit := range bits {
			combiner.Put(i, bit&1 != 0)
			if i == worst {
				computed[0] = bit&1 != 0
			}
			if i == worst1 {
				computed[1] = bit&1 != 0
			}
		}
		return combiner, computed
	}

	if len(f.mapping) > l && f.mapping[l] > 0 {
		if f.premodulo[l] != 0 {
			in = SingleValue(hash.Hash(in.Feature(0), 0, f.premodulo[l]))
		}
		var val = f.layers[l][0].Forward(in.Feature(0), false)
		//println(in.Feature(0), "=>", val)
		return SingleValue(val), []bool{false, false}
	} else {
		if f.premodulo[l] != 0 {
			in = SingleValue(hash.Hash(in.Feature(0), 0, f.premodulo[l]))
		}
		var bit = f.layers[l][0].Forward(in.Feature(0), false)
		return SingleValue(bit & 1), []bool{(bit & 1) != 0, (bit & 1) != 0}
	}

	return nil, []bool{false, false}
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

type tally4io struct {
	io    FeedforwardNetworkParityInOutput
	shift int
}

func (io tally4io) Feature(n int) uint32 {
	return (uint32(io.io.Output()) ^ uint32(io.io.Parity())) >> (n * io.shift)
}

// Tally4 tallies the network on ParityInOutput, tuning the worst-th hashtron
// in the network f storing data in tally. Loss can be nil if predicting power of 2 classes,
// or an actual minus expected difference or any other loss (0 means correct).
func (f *FeedforwardNetwork) Tally4(io FeedforwardNetworkParityInOutput, worst int, tally datasets.AnyTally,
	loss func(actual, expected, mask uint32) uint32) {
	if loss == nil {
		loss = func(actual, expected, mask uint32) uint32 {
			if actual >= expected {
				return actual - expected
			}
			return expected - actual
		}
	}
	if f.GetLastCells() > 1 {
		f.tally(io, tally4io{io: io, shift: 1}, worst, tally, func(i, j FeedforwardNetworkInput) bool {
			var ifeat, jfeat uint32
			for k := byte(0); k < f.GetLastCells(); k++ {
				ifeat |= (i.Feature(int(k)) & 1) << k
				jfeat |= (j.Feature(int(k)) & 1) << k
			}
			//println(ifeat, jfeat)
			return loss(ifeat, jfeat, (1<<f.GetLastCells()))>>1 != 0
		})
		return
	}
	mask := uint32(uint32(1<<f.GetBits()) - 1)
	out := uint32(io.Output()^io.Parity()) & mask
	f.tally2(io, tally4io{io: io, shift: 0}, worst, tally, func(i FeedforwardNetworkInput) uint32 {
		ifm := (i.Feature(0)) & mask
		//println(ifm, out)
		return loss(ifm, out, mask)
	})
}

// Tally3 tallies the network like Tally2, except it can also balance the dataset using input parity bit.
// Loss is 0 if the output is correct, below or equal to maxloss otherwise.
func (f *FeedforwardNetwork) tally3(in FeedforwardNetworkParityInput, output FeedforwardNetworkInput,
	worst int, tally datasets.AnyTally, loss func(i FeedforwardNetworkInput) uint32) {
	if in.Parity() {
		output = tally3io{
			par: in,
			out: output,
		}
	}
	f.tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
		return loss(i) < loss(j)
	})
}

// Tally2 tallies the network like Tally, except it can also optimize n-way classifiers. Loss is 0 if the
// output is correct, below or equal to maxloss otherwise.
func (f *FeedforwardNetwork) tally2(in, output FeedforwardNetworkInput, worst int, tally datasets.AnyTally,
	loss func(i FeedforwardNetworkInput) uint32) {
	l := f.GetLayer(worst)
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		f.tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
			//return (loss(i) == 0) && (loss(j) > 0)
			return loss(i) < loss(j)
		})
		return
	}

	f.tally(in, output, worst, tally, func(i, j FeedforwardNetworkInput) bool {
		//return (loss(i) == 0) && (loss(j) > 0)
		return loss(i) < loss(j)
	})
	return
}

// Tally tallies the network on input/output pair with respect to to-be-trained worst hashtron.
// The tally is stored into thread safe structure Tally. Two ouputs i, j can be compared to be less
// worse using the function less (returning true if output i is less worse than output j).
func (f *FeedforwardNetwork) tally(in, output FeedforwardNetworkInput, worst int, tally datasets.AnyTally, less func(i, j FeedforwardNetworkInput) bool) {

	l := f.GetLayer(worst)
	origin := in
	preadd_input := in
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		in := Intermediate(&inferPreaddBase{add: SingleValue(0), in: in, base: 0})

		var predicted [2]FeedforwardNetworkInput
		var compute [2]int8

		for l_prev := 0; l_prev < l; l_prev += 2 {
			if f.preadd[l_prev] == preAddition {
				in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l_prev)}
			}
			if f.preadd[l_prev] == interAddition {
				in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
			}
			preadd_input = in
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifw := in.Feature(f.GetPosition(worst))
		if f.preadd[l] == interAddition {
			ifw += preadd_input.Feature(f.GetPosition(worst))
		} else if f.preadd[l] == preAddition {
			ifw += origin.Feature(f.GetFrontOffset(l) + f.GetPosition(worst))
		}
		if f.premodulo[l] != 0 {
			ifw = hash.Hash(ifw, uint32(f.GetPosition(worst)), f.premodulo[l])
		}
		if tally.IsGlobalPremodulo() {
			spm := tally.GetGlobalSaltPremodulo()
			ifw = hash.Hash(ifw, spm[0], spm[1])
		}
		for neg := 0; neg < 2; neg++ {
			inter := in
			if f.preadd[l] == preAddition {
				inter = &inferPreaddBase{add: origin, in: inter, base: f.GetFrontOffset(l)}
			}
			if f.preadd[l] == interAddition {
				inter = &inferPreaddBase{add: preadd_input, in: inter, base: 0}
			}
			preadd_input_next := inter
			inter, computed := f.Forward(inter, l, f.GetPosition(worst), neg)
			if computed[0] {
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
				if f.preadd[l_post] == preAddition {
					inter = &inferPreaddBase{add: origin, in: inter, base: f.GetFrontOffset(l_post)}
				}
				if f.preadd[l_post] == interAddition {
					inter = &inferPreaddBase{add: preadd_input_next, in: inter, base: 0}
				}
				preadd_input_next = inter
				inter, _ = f.Forward(inter, l_post, -1, 0)
			}
			predicted[neg] = inter
		}
		if !less(predicted[0], output) && !less(predicted[1], output) &&
			!less(output, predicted[0]) && !less(output, predicted[1]) {
			// we are correct anyway

			return
		}

		if !less(predicted[0], predicted[1]) && !less(predicted[1], predicted[0]) {
			// can't change

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
		return
	}
	if len(f.mapping) > l && f.mapping[l] > 0 {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			if f.preadd[l_prev] == preAddition {
				in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l_prev)}
			}
			if f.preadd[l_prev] == interAddition {
				in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
			}
			preadd_input = in
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint32(in.Feature(0))
		if f.premodulo[l] != 0 {
			ifeature = hash.Hash(uint32(ifeature), 0, f.premodulo[l])
		}
		if tally.IsGlobalPremodulo() {
			spm := tally.GetGlobalSaltPremodulo()
			ifeature = hash.Hash(uint32(ifeature), spm[0], spm[1])
		}
		if f.GetBits() == 1 {
			if f.preadd[l] == preAddition {
				in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l)}
			}
			if f.preadd[l] == interAddition {
				in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
			}
			preadd_input = in
			_, actual := f.Forward(in, l, f.GetPosition(worst), 0)
			changed := actual[0] != (output.Feature(0)&1 != 0)
			tally.AddToCorrect(ifeature, 2*int8(output.Feature(0)&1)-1, changed)
		} else {
			tally.AddToMapping(uint16(ifeature), uint64(output.Feature(0)))
		}
	} else {
		for l_prev := 0; l_prev < l; l_prev += 2 {
			if f.preadd[l_prev] == preAddition {
				in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l_prev)}
			}
			if f.preadd[l_prev] == interAddition {
				in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
			}
			preadd_input = in
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifeature := uint32(in.Feature(0))
		if f.premodulo[l] != 0 {
			ifeature = hash.Hash(uint32(ifeature), 0, f.premodulo[l])
		}
		if tally.IsGlobalPremodulo() {
			spm := tally.GetGlobalSaltPremodulo()
			ifeature = hash.Hash(uint32(ifeature), spm[0], spm[1])
		}
		if f.preadd[l] == preAddition {
			in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l)}
		}
		if f.preadd[l] == interAddition {
			in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
		}
		preadd_input = in
		_, actual := f.Forward(in, l, f.GetPosition(worst), 0)
		changed := actual[0] != (output.Feature(0)&1 != 0)
		tally.AddToCorrect(ifeature, 2*int8(output.Feature(0)&1)-1, changed)
	}
}

// GetBits reports the number of bits predicted by this network
func (f *FeedforwardNetwork) GetBits() (ret byte) {
	if len(f.mapping) == 0 {
		return 1
	}
	ret = f.mapping[len(f.mapping)-1]
	if ret == 0 {
		ret = 1
	}
	return
}

// GetLastCells gets last cells
func (f *FeedforwardNetwork) GetLastCells() (ret byte) {
	if len(f.layers) == 0 {
		return 0
	}
	ret = byte(len(f.layers[len(f.layers)-1]))
	if ret == 0 && len(f.layers) >= 2 {
		ret = byte(len(f.layers[len(f.layers)-2]))
	}
	return
}

// GetBits reports the number of classes predicted by this network
func (f *FeedforwardNetwork) GetClasses() (ret uint16) {
	ret = uint16(1 << f.GetBits())
	ret2 := uint16(1 << f.GetLastCells())
	if ret2 > ret {
		return ret2
	}
	return
}

func (f *FeedforwardNetwork) AnyTally(io FeedforwardNetworkParityInOutput, worst []int, t datasets.AnyTally,
	loss func(actual, expected, mask uint32) uint32) {
	if len(worst) == 1 {
		f.Tally4(io, worst[0], t, loss)
	} else if len(worst) == 2 {
		f.PreTally4(io, [2]int{worst[0], worst[1]}, t, loss)
	}
}

// PreTally4 preTallys two worst hashtrons together for parity-based outputs
// Similar to preTally2 but handles parity input/output and power-of-2 classification
func (f *FeedforwardNetwork) PreTally4(io FeedforwardNetworkParityInOutput, worst [2]int, dist datasets.AnyTally,
	loss func(actual, expected, mask uint32) uint32) {

	// Default loss function if nil (absolute difference)
	if loss == nil {
		loss = func(actual, expected, mask uint32) uint32 {
			if actual >= expected {
				return actual - expected
			}
			return expected - actual
		}
	}

	// Handle multi-cell output case
	if f.GetLastCells() > 1 {
		f.preTally(io, tally4io{io: io, shift: 1}, worst, dist, func(i, j FeedforwardNetworkInput) bool {
			var ifeat, jfeat uint32
			for k := byte(0); k < f.GetLastCells(); k++ {
				ifeat |= (i.Feature(int(k)) & 1) << k
				jfeat |= (j.Feature(int(k)) & 1) << k
			}
			return loss(ifeat, jfeat, (1<<f.GetLastCells())-1)>>1 != 0
		})
		return
	}

	// Single-cell output case
	mask := uint32(1<<f.GetBits() - 1)
	out := uint32(io.Output()^io.Parity()) & mask
	f.preTally2(io, tally4io{io: io, shift: 0}, worst, dist, func(i FeedforwardNetworkInput) uint32 {
		ifm := i.Feature(0) & mask
		return loss(ifm, out, mask)
	})
}

// preTally2 preTallys two worst hashtrons together using a loss function
// Similar to preTally but uses a loss function instead of less comparator
// Loss is 0 if output is correct, higher values indicate worse outputs
func (f *FeedforwardNetwork) preTally2(in, output FeedforwardNetworkInput, worst [2]int, dist datasets.AnyTally,
	loss func(i FeedforwardNetworkInput) uint32) {

	// For networks with combiners, use the full comparison version
	l0 := f.GetLayer(worst[0])
	if len(f.combiners) > l0+1 && f.combiners[l0+1] != nil {
		f.preTally(in, output, worst, dist, func(i, j FeedforwardNetworkInput) bool {
			return loss(i) < loss(j)
		})
		return
	}

	// For simple cases, use optimized version
	f.preTally(in, output, worst, dist, func(i, j FeedforwardNetworkInput) bool {
		// Prefer outputs with lower loss
		return loss(i) < loss(j)

		// Alternative strict version (uncomment if preferred):
		// return (loss(i) == 0) && (loss(j) > 0)
	})
}

// preTally preTallys two worst hashtrons together using negation-based override system
func (f *FeedforwardNetwork) preTally(in, output FeedforwardNetworkInput, worst [2]int, dist datasets.AnyTally, less func(i, j FeedforwardNetworkInput) bool) {
	l := f.GetLayer(worst[0])
	l2 := f.GetLayer(worst[1])
	origin := in
	preadd_input := in
	if len(f.combiners) > l+1 && f.combiners[l+1] != nil {
		in := Intermediate(&inferPreaddBase{add: SingleValue(0), in: in, base: 0})

		var predicted [4]FeedforwardNetworkInput
		var compute [2][4]bool
		var ifws [4]uint32

		for l_prev := 0; l_prev < l; l_prev += 2 {
			if f.preadd[l_prev] == preAddition {
				in = &inferPreaddBase{add: origin, in: in, base: f.GetFrontOffset(l_prev)}
			}
			if f.preadd[l_prev] == interAddition {
				in = &inferPreaddBase{add: preadd_input, in: in, base: 0}
			}
			preadd_input = in
			in, _ = f.Forward(in, l_prev, -1, 0)
		}
		ifw := in.Feature(f.GetPosition(worst[0]))
		if f.preadd[l] == interAddition {
			ifw += preadd_input.Feature(f.GetPosition(worst[0]))
		} else if f.preadd[l] == preAddition {
			ifw += origin.Feature(f.GetFrontOffset(l) + f.GetPosition(worst[0]))
		}
		if f.premodulo[l] != 0 {
			ifw = hash.Hash(ifw, uint32(f.GetPosition(worst[0])), f.premodulo[l])
		}
		if dist.IsGlobalPremodulo() {
			spm := dist.GetGlobalSaltPremodulo()
			ifw = hash.Hash(ifw, spm[0], spm[1])
		}
		if l == l2 {
			ifws[0] = in.Feature(f.GetPosition(worst[1]))
			if f.preadd[l] == interAddition {
				ifws[0] += preadd_input.Feature(f.GetPosition(worst[1]))
			} else if f.preadd[l] == preAddition {
				ifws[0] += origin.Feature(f.GetFrontOffset(l) + f.GetPosition(worst[1]))
			}
			if f.premodulo[l] != 0 {
				ifws[0] = hash.Hash(ifws[0], uint32(f.GetPosition(worst[1])), f.premodulo[l])
			}
			if dist.IsGlobalPremodulo() {
				spm := dist.GetGlobalSaltPremodulo()
				ifws[0] = hash.Hash(ifws[0], spm[0], spm[1])
			}
			ifws[1] = ifws[0]
			ifws[2] = ifws[0]
			ifws[3] = ifws[0]
		}
		for neg := 0; neg < 4; neg++ {
			inter := in
			if f.preadd[l] == preAddition {
				inter = &inferPreaddBase{add: origin, in: inter, base: f.GetFrontOffset(l)}
			}
			if f.preadd[l] == interAddition {
				inter = &inferPreaddBase{add: preadd_input, in: inter, base: 0}
			}
			preadd_input_next := inter
			var computed []bool
			inter, computed = f.Forward(inter, l, f.GetPosition(worst[0]), neg&1, f.GetPosition(worst[1]), neg>>1)
			if computed[0] {
				compute[0][neg] = true
			} else {
				compute[0][neg] = false
			}
			if l == l2 {
				if computed[1] {
					compute[1][neg] = true
				} else {
					compute[1][neg] = false
				}
			}
			if neg == 0 {
				if inter.Disregard(f.GetLayerPosition(l, worst[0])) {
					return
				}
			}
			for l_post := l + 2; l_post < f.LenLayers(); l_post += 2 {
				preadd_input_current := preadd_input_next // Save before modification
				if l_post == l2 {
					ifws[neg] = inter.Feature(f.GetPosition(worst[1]))
					if f.preadd[l_post] == interAddition {
						ifws[neg] += preadd_input_current.Feature(f.GetPosition(worst[1]))
					} else if f.preadd[l_post] == preAddition {
						ifws[neg] += origin.Feature(f.GetFrontOffset(l_post) + f.GetPosition(worst[1]))
					}
					if f.premodulo[l_post] != 0 {
						ifws[neg] = hash.Hash(ifws[neg], uint32(f.GetPosition(worst[1])), f.premodulo[l_post])
					}
					if dist.IsGlobalPremodulo() {
						spm := dist.GetGlobalSaltPremodulo()
						ifws[neg] = hash.Hash(ifws[neg], spm[0], spm[1])
					}
				}
				if f.preadd[l_post] == preAddition {
					inter = &inferPreaddBase{add: origin, in: inter, base: f.GetFrontOffset(l_post)}
				}
				if f.preadd[l_post] == interAddition {
					inter = &inferPreaddBase{add: preadd_input_next, in: inter, base: 0}
				}
				preadd_input_next = inter
				if l_post != l2 {
					inter, _ = f.Forward(inter, l_post, -1, 0)
				} else {
					var computed []bool
					inter, computed = f.Forward(inter, l_post, f.GetPosition(worst[1]), neg>>1)
					if computed[0] {
						compute[1][neg] = true
					} else {
						compute[1][neg] = false
					}
				}
			}
			predicted[neg] = inter
		}

		//fmt.Println(compute)

		// initially we inspect if we are correct anyway
		if !less(predicted[0], output) && !less(output, predicted[0]) &&
			!less(predicted[1], output) && !less(output, predicted[1]) &&
			!less(predicted[2], output) && !less(output, predicted[2]) &&
			!less(predicted[3], output) && !less(output, predicted[3]) {
			return
		}

		irrelevantB := !less(predicted[0], predicted[2]) && !less(predicted[2], predicted[0]) &&
			!less(predicted[1], predicted[3]) && !less(predicted[3], predicted[1])

		if irrelevantB {
			// (1) B is irrelevant — we’ve already checked that
			if compute[0][0] != compute[0][2] || compute[0][1] != compute[0][3] {
				panic("B was in fact relevant")
			}

			var aShouldBe, ambiguous bool
			if compute[0][0] == compute[0][1] { // constant neuron
				aShouldBe = compute[0][0]
			} else {
				// check which predicted matches output
				pred0_matches := !less(predicted[0], output) && !less(output, predicted[0])
				pred1_matches := !less(predicted[1], output) && !less(output, predicted[1])

				if pred0_matches {
					aShouldBe = compute[0][0]
				} else if pred1_matches {
					aShouldBe = compute[0][1]
				} else {
					ambiguous = true
				}
			}

			if !ambiguous {
				// Then write (respecting any prior decision):
				val0, exists0 := dist.GetCellDecision(0, ifw)
				if !exists0 || val0 == aShouldBe {
					dist.SetCellDecision(0, ifw, aShouldBe)
					return
				}
			}
		}

		irrelevantA := !less(predicted[0], predicted[1]) && !less(predicted[1], predicted[0]) &&
			!less(predicted[2], predicted[3]) && !less(predicted[3], predicted[2])

		if irrelevantA {
			if compute[1][0] != compute[1][1] || compute[1][2] != compute[1][3] {

				if ifws[0] == ifws[2] {

					var ifwreal uint32
					var bShouldBe, ambiguous bool
					if compute[1][0] == compute[1][2] { // constant neuron
						bShouldBe = compute[1][0]
						ifwreal = ifws[0]
					} else {
						// check which predicted matches output
						pred0_matches := !less(predicted[0], output) && !less(output, predicted[0])
						pred2_matches := !less(predicted[2], output) && !less(output, predicted[2])

						if pred0_matches {
							bShouldBe = compute[1][0]
							ifwreal = ifws[0]
						} else if pred2_matches {
							bShouldBe = compute[1][2]
							ifwreal = ifws[2]
						} else {
							ambiguous = true
						}
					}

					if !ambiguous {
						// Then write (respecting any prior decision):
						val1, exists1 := dist.GetCellDecision(1, ifwreal)
						if !exists1 || val1 == bShouldBe {
							dist.SetCellDecision(1, ifwreal, bShouldBe)
							return
						}
					}

				} else {
					var ifwreal = [2]uint32{ifws[0], ifws[2]}
					var bShouldBe = [2]bool{compute[1][0], compute[1][2]}

					// Then write (respecting any prior decision):
					val10, exists10 := dist.GetCellDecision(1, ifwreal[0])
					val11, exists11 := dist.GetCellDecision(1, ifwreal[1])
					if !exists10 || val10 == bShouldBe[0] {
						dist.SetCellDecision(1, ifwreal[0], bShouldBe[0])
					}
					if !exists11 || val11 == bShouldBe[1] {
						dist.SetCellDecision(1, ifwreal[1], bShouldBe[1])
					}
					if !exists10 || val10 == bShouldBe[0] || !exists11 || val11 == bShouldBe[1] {
						return
					}
				}

			} else {
				// Neuron A truly irrelevant, compute values equal — choose any consistently:
				bShouldBe := compute[1][0]
				ifwreal := ifws[0]

				val1, exists1 := dist.GetCellDecision(1, ifwreal)
				if !exists1 || val1 == bShouldBe {
					dist.SetCellDecision(1, ifwreal, bShouldBe)
					return
				}
			}

		}

		// Case 3: Need joint training (original logic)
		for neg := 0; neg < 4; neg++ {
			if !less(predicted[neg], output) && !less(output, predicted[neg]) {
				val0, exists0 := dist.GetCellDecision(0, ifw)
				val1, exists1 := dist.GetCellDecision(1, ifws[neg])

				if (!exists0 || val0 == compute[0][neg]) &&
					(!exists1 || val1 == compute[1][neg]) {
					dist.SetCellDecision(0, ifw, compute[0][neg])
					dist.SetCellDecision(1, ifws[neg], compute[1][neg])
					return
				}
			}
		}
		return
	}
}
