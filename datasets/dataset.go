// Package datasets implements the Neurlang dataset type
package datasets

import "math/rand"
import "sort"

// Dataset holds keys mapped to booleans
type Dataset map[uint32]bool

// Init initializes the Dataset
func (d *Dataset) Init() {
	*d = make(map[uint32]bool)
}

// Split splits dataset into a true set and a false set
func (d Dataset) Split() (o SplittedDataset) {
	o[0] = make(map[uint32]struct{})
	o[1] = make(map[uint32]struct{})
	for k, v := range d {
		if v {
			o[1][k] = struct{}{}
		} else {
			o[0][k] = struct{}{}
		}
	}
	return
}

// Datamap holds keys mapped to values
type Datamap map[uint16]uint64

// Init initializes the Datamap
func (d *Datamap) Init() {
	*d = make(map[uint16]uint64)
}

// Split splits datamap into a true set and a false set
func (d Datamap) Split() (o SplittedDataset) {
	o[0] = make(map[uint32]struct{})
	o[1] = make(map[uint32]struct{})
	var bits uint64
	for _, v := range d {
		bits |= v
	}
	for k, v := range d {
		for i := uint16(0); (1 << i) < bits; i++ {
			if (v>>i)&1 == 1 {
				o[1][uint32(k)|uint32(i)<<16] = struct{}{}
			} else {
				o[0][uint32(k)|uint32(i)<<16] = struct{}{}
			}
		}
	}
	return
}

// Reduce reduces the datamap
func (d Datamap) Reduce(whole bool) (o Datamap) {
	o = make(Datamap)
	var exists = make(map[uint64]struct{})
	var arr []uint64
	for _, v := range d {
		if _, ok := exists[v]; ok {
			continue
		}
		arr = append(arr, v)
		exists[v] = struct{}{}
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i] < arr[j]
	})
	if whole {
		for k, v := range d {
			for i := range arr {
				if arr[i] == v {
					o[k] = uint64(i)
					break
				}
			}
		}
	} else {
		for i, v := range arr {
			o[uint16(i)] = v
		}
	}
	return
}

// Splitter is a dataset that can be split into a SplittedDataset
type Splitter interface {
	Split() (o SplittedDataset)
}

// SplittedDataset is a dataset that has been split into the false set and true set
type SplittedDataset [2]map[uint32]struct{}

// Init initializes a SplittedDataset
func (d *SplittedDataset) Init() {
	d[0] = make(map[uint32]struct{})
	d[1] = make(map[uint32]struct{})
}

// Split splits SplittedDataset into a true set and a false set
func (d SplittedDataset) Split() (o SplittedDataset) {
	return d
}

// BalanceDataset fills the smaller set with random number until it matches the bigger set
func BalanceDataset(d SplittedDataset) SplittedDataset {
	if len(d[0]) == len(d[1]) {
		return d
	}
	for len(d[0]) < len(d[1]) {
		var w = uint32(uint16(rand.Uint32()))
		if _, ok := d[1][w]; !ok {
			d[0][w] = struct{}{}
		}
	}
	for len(d[1]) < len(d[0]) {
		var w = uint32(uint16(rand.Uint32()))
		if _, ok := d[0][w]; !ok {
			d[1][w] = struct{}{}
		}
	}
	return d
}
