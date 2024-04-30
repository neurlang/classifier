// Package datasets implements the Neurlang dataset type
package datasets

import "math/rand"

type Splitter interface {
	Split() (o SplittedDataset)
}

type Dataset map[uint32]bool

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

type Datamap map[uint16]uint16

func (d *Datamap) Init() {
	*d = make(map[uint16]uint16)
}

// Split splits datamap into a true set and a false set
func (d Datamap) Split() (o SplittedDataset) {
	o[0] = make(map[uint32]struct{})
	o[1] = make(map[uint32]struct{})
	var bits uint16
	for _, v := range d {
		bits |= v
	}
	for k, v := range d {
		for i := uint16(0); (1<<i) < bits; i++ {
			if (v >> i) & 1 == 1 {
				o[1][uint32(k) | uint32(1) << (i+16)] = struct{}{}
			} else {
				o[0][uint32(k) | uint32(1) << (i+16)] = struct{}{}
			}
		}
	}
	return
}



type SplittedDataset [2]map[uint32]struct{}

// BalanceDataset fills the smaller set with random number until it matches the bigger set
func BalanceDataset(d SplittedDataset) SplittedDataset {
	if len(d[0]) == len(d[1]) {
		return d
	}
	for len(d[0]) < len(d[1]) {
		var w = rand.Uint32()
		for _, ok := d[1][w]; !ok; w++ {
			d[0][w] = struct{}{}
			if len(d[1]) == len(d[0]) {
				break
			}
		}
	}
	for len(d[1]) < len(d[0]) {
		var w = rand.Uint32()
		for _, ok := d[0][w]; !ok; w++ {
			d[1][w] = struct{}{}
			if len(d[1]) == len(d[0]) {
				break
			}
		}
	}
	return d
}
