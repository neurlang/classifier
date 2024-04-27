// Package datasets implements the Neurlang dataset type
package datasets

import "math/rand"

type Dataset map[uint32]bool

func (d *Dataset) Init() {
	*d = make(map[uint32]bool)
}

type SplittedDataset [2]map[uint32]struct{}

// SplitDataset splits dataset into a true set and a false set
func SplitDataset(d Dataset) (o SplittedDataset) {
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

// BalanceDataset fills the smaller set with random number until it matches the bigger set
func BalanceDataset(d SplittedDataset) SplittedDataset {
	if len(d[0]) == len(d[1]) {
		return d
	}
	for len(d[0]) < len(d[1]) {
		var w = rand.Uint32()
		if _, ok := d[1][w]; !ok {
			d[0][w] = struct{}{}
		}
	}
	for len(d[1]) < len(d[0]) {
		var w = rand.Uint32()
		if _, ok := d[0][w]; !ok {
			d[1][w] = struct{}{}
		}
	}
	return d
}
