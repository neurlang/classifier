package main

import "github.com/neurlang/classifier/hashtron"
import "sync"

type Intermediate interface {
	Feature(n int) uint32
	Dropout(n int) bool
}

// SingleValue

type SingleValue uint32

func (v SingleValue) Feature(n int) uint32 {
	return uint32(v)
}
func (v SingleValue) Dropout(n int) bool {
	return false
}

// Grid

type Grid struct {
	g []bool
	num byte
}

func NewGrid(num byte, repeat byte) (*Grid) {
	n := int(num)
	r := int(repeat)
	g := make([]bool, n*n*n*n*n*n * r)
	return &Grid{
		g, num,
	}
}

func (g *Grid) Put(n int, v uint64) {
	(g.g)[n] = (v & 1) != 0
}

func (g *Grid) Feature(n int) (o uint32) {
	num := int(g.num)
	numnum := num*num
	r := (n / numnum) / numnum
	y := (n / numnum) % numnum
	x := n % numnum
	for i := 0; i < num; i++ {
	for j := 0; j < num; j++ {
		o <<= 1
		if (g.g)[r * numnum * numnum + numnum*(num*y+i)+(num*x+j)] {
			o |= 1
		}
	}
	}
	return o
}
func (s *Grid) Dropout(n int) bool {
	return false
}


// SumPool

type SumPool struct {
	s []bool
	num byte
	num2 byte
}

func NewSumPool(num, num2, repeat byte) (*SumPool) {
	n := int(num)
	m := int(num2)
	r := int(repeat)
	s := make([]bool, m*m*n*n*r)
	return &SumPool{
		s, num, num2,
	}
}

func (s *SumPool) Put(n int, v uint64) {
	num := int(s.num)
	num2 := int(s.num2)
	numnum2 := num*num2
	numnum := num*num
	r := (n / numnum2) / numnum2
	x := (n / numnum2) % numnum2
	y := n % numnum2
	xx := x / num
	xy := x % num
	yx := y / num
	yy := y % num
	(s.s)[r * numnum2 * numnum2 + numnum*(xx+num2*yx) + (xy+num*yy)] = (v & 1) != 0
}
func (s *SumPool) Dropout(n int) bool {
	if n < 0 {
		return false
	}
	num := int(s.num)
	num2 := int(s.num2)
	numnum2 := num*num2
	numnum := num*num
	r := (n / numnum2) / numnum2
	x := (n / numnum2) % numnum2
	y := n % numnum2
	xx := x / num
	yx := y / num
	var w = 0
	for i := 0; i < num; i++ {
	for j := 0; j < num; j++ {
		if (s.s)[r * numnum2 * numnum2 + numnum*(num2*xx+yx)+(num*i+j)] {
			w++
		} else {
			w--
		}
	}}
	return !(w == -1 || w == +1)
}
func (s *SumPool) Feature(m int) (o uint32) {
	num := int(s.num)
	num2 := int(s.num2)
	num2num2 := num2*num2
	numnum := num*num
	m /= num2num2
	m /= numnum
	for n := 0; n < num2num2; n++ {
		var w = 0
		for i := 0; i < numnum; i++ {
			if (s.s)[num2num2*numnum*m + numnum*(n)+(i)] {
				w++
			} else {
				w--
			}
		}
		if w > 0 {
			o |= 1 << n
		}
	}
	return 
}

type FeedforwardNetworkInput interface {
	Feature(n int) uint32
}

type FeedforwardNetwork struct {
	layers [][]hashtron.Hashtron
	grids []*[2]byte
	sumpools []*[3]byte
	mapping []bool
}

func (f FeedforwardNetwork) Len() (o int) {
	for _, v := range f.layers {
		o += len(v)
	}
	return
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
	f.grids = append(f.grids, nil)
	f.sumpools = append(f.sumpools, nil)
	f.mapping = append(f.mapping, bits > 1)
}

func (f *FeedforwardNetwork) NewGrid(bits, repeat byte) {
	f.layers = append(f.layers, nil)
	f.grids = append(f.grids, &[2]byte{bits, repeat})
	f.sumpools = append(f.sumpools, nil)
	f.mapping = append(f.mapping, false)
}
func (f *FeedforwardNetwork) NewSumPool(bits, bits2, repeat byte) {
	f.layers = append(f.layers, nil)
	f.grids = append(f.grids, nil)
	f.sumpools = append(f.sumpools, &[3]byte{bits, bits2, repeat})
	f.mapping = append(f.mapping, false)
	
}

func (f FeedforwardNetwork) IsMapLayerOf(n int) (bool) {
	if f.GetLayer(n) == -1 {
		return false
	}
	return f.mapping[f.GetLayer(n)]
}
func (f FeedforwardNetwork) Forward(in FeedforwardNetworkInput, l, worst, neg int) (inter Intermediate, computed bool) {
	if len(f.grids) > l+1 && f.grids[l+1] != nil {
		var gridnum = *f.grids[l+1]
		var grid = NewGrid(gridnum[0], gridnum[1])
		wg := sync.WaitGroup{}
		for i := range f.layers[l] {
			wg.Add(1)
			go func(i int) {
				var bit = f.layers[l][i].Forward(in.Feature(i), (i == worst) && (neg == 1))
				grid.Put(i, bit)
				if i == worst {
					computed = bit & 1 != 0
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
		return grid, computed
	}
	if len(f.sumpools) > l+1 && f.sumpools[l+1] != nil {
		var sumpoolnum = *f.sumpools[l+1]
		var sumpool = NewSumPool(sumpoolnum[0], sumpoolnum[1], sumpoolnum[2])
		//var sumpool = new(SumPool4x3)
		wg := sync.WaitGroup{}
		for i := range f.layers[l] {
			wg.Add(1)
			go func(i int) {
				var bit = f.layers[l][i].Forward(in.Feature(i), (i == worst) && (neg == 1))
				sumpool.Put(i, bit)
				if i == worst {
					computed = bit & 1 != 0
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
		return sumpool, computed
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
