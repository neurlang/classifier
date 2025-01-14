// Package isvirus implements the IsVirus TLSH file Hashes machine learning Dataset
package isvirus

import "github.com/neurlang/classifier/datasets/stringhash"

type Dataslice struct {}

func (_ Dataslice) Get(n int) Sample {
	if len(virus) <= n {
		n -= len(virus)
		return Sample(stringhash.Sample{
			Str: clean[n],
			Out: 0,
		})
	}
	return Sample(stringhash.Sample{
		Str: virus[n],
		Out: 1,
	})
}
func (_ Dataslice) Len() int {
	return len(virus) + len(clean)
}

type Sample stringhash.Sample
type ByteSample stringhash.ByteSample

func (s Sample) Balance() stringhash.BalancedSample {
	return stringhash.BalancedSample{
		Str: s.Str,
		Out: s.Out,
	}
}
func (s Sample) Byte() ByteSample {
	return ByteSample(stringhash.ByteSample{
		Buf: []byte(s.Str),
		Out: s.Out,
	})
}
func (s ByteSample) Balance() stringhash.BalancedByteSample {
	return stringhash.BalancedByteSample{
		Buf: s.Buf,
		Out: s.Out,
	}
}
