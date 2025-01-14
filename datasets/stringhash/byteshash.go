package stringhash

import "github.com/neurlang/classifier/hash"

type ByteSample struct {
	Buf []byte
	Out uint16
}

func (s ByteSample) Feature(n int) uint32 {
	var ret uint32
	for j := 0; j < 4; j++ {
		ret ^= uint32(s.Buf[hash.Hash(uint32(n), uint32(j), uint32(len(s.Buf)))]) << uint32(8*j)
	}
	return ret
}
func (s ByteSample) Parity() uint16 {
	return 0
}
func (s ByteSample) Output() uint16 {
	return s.Out
}

type BalancedByteSample struct {
	Buf []byte
	Out uint16
}

func (s BalancedByteSample) Feature(n int) uint32 {
	var ret uint32
	for j := 0; j < 4; j++ {
		ret ^= uint32(s.Buf[hash.Hash(uint32(n), uint32(j), uint32(len(s.Buf)))]) << uint32(8*j)
	}
	return ret
}
func (s BalancedByteSample) Parity() uint16 {
	var ret uint32
	for _, b := range s.Buf {
		ret = hash.Hash(ret, uint32(b), 0xFFFFFFFF)
	}
	return uint16(ret)
}
func (s BalancedByteSample) Output() uint16 {
	return s.Out
}
