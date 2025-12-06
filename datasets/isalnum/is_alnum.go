// Package isalnum implements the IsAlnum Dataset
package isalnum

import "github.com/neurlang/classifier/datasets"

type Dataslice struct{}

func (d Dataslice) Get(n int) Sample {
	return Sample(n)
}
func (d Dataslice) Len() int {
	return 256
}

// Set materializes the Dataslice, taking only feature 0 into account
func (d Dataslice) Set() (set datasets.Dataset) {
	set.Init()
	// Loop through ASCII characters
	for i := 0; i < d.Len(); i++ {
		set[d.Get(i).Feature(0)] = d.Get(i).Output()^d.Get(i).Parity() != 0
	}
	return
}

type Sample rune

func (c Sample) Feature(_ int) uint32 {
	return uint32(c)
}

func (c Sample) Parity() uint16 {
	return 0
}

func (c Sample) Output() uint16 {
	if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
		return 1
	}
	return 0
}
