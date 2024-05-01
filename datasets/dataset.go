// Package datasets implements the Neurlang dataset type
package datasets

import "math/rand"
import "sort"

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

// SplitN splits datamap into a set for each equivalence group
func (d Datamap) SplitN() (o SplittedNDataset) {
	var used = make(map[uint16]int)
	for k, v := range d {
		if n, ok := used[v]; ok {
			o[n][uint32(k)] = struct{}{}
			continue
		}
		used[v] = len(o)
		var m = make(map[uint32]struct{})
		m[uint32(k)] = struct{}{}
		o = append(o, m)
	}
	return
}

// Reduce reduces the datamap
func (d Datamap) Reduce(whole bool) (o Datamap) {
	o = make(Datamap)
	var exists = make(map[uint16]struct{})
	var arr []uint16
	for _, v := range d {
		if _, ok := exists[v]; ok {
			continue
		}
		arr = append(arr, v)
		exists[v] = struct{}{}
	}
	sort.Slice(arr, func (i, j int) bool {
		return arr[i] < arr[j]
	})
	if whole {
		for k, v := range d {
			for i := range arr {
				if arr[i] == v {
					o[k] = uint16(i)
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

type Splitter interface {
	Split() (o SplittedDataset)
}

type SplittedDataset [2]map[uint32]struct{}

type SplittNer interface {
	SplitN() (o SplittedNDataset)
}

type SplittedNDataset []map[uint32]struct{}

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

// BalanceDatasetN fills the smaller set with random number until it matches the bigger set
func BalanceDatasetN(d SplittedNDataset) SplittedNDataset {
	var biggest int
	var different bool
	for j := range d {
		if len(d[j]) > biggest {
			biggest = len(d[j])
		}
		if len(d[j]) != len(d[0]) {
			different = true
		}
	}

	if !different {
		return d
	}
	for j := range d {
	outer:
		for len(d[j]) < biggest {
			var w = rand.Uint32()
			for i := range d {
				if i == j {
					continue
				}
				if _, ok := d[i][w]; ok {
					continue outer
				}
			}
			d[j][w] = struct{}{}
		}
	}
	return d
}
