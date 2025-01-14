package stringhash

import "github.com/neurlang/classifier/hash"

type Sample struct {
	Str string
	Out uint16
}

func (s Sample) Feature(n int) uint32 {
	return hash.StringHash(uint32(n), s.Str)
}
func (s Sample) Parity() uint16 {
	return 0
}
func (s Sample) Output() uint16 {
	return s.Out
}

type BalancedSample struct {
	Str string
	Out uint16
}

func (s BalancedSample) Feature(n int) uint32 {
	return hash.StringHash(uint32(n), s.Str)
}
func (s BalancedSample) Parity() uint16 {
	return uint16(hash.StringHash(0xFFFFFFFF, s.Str))
}
func (s BalancedSample) Output() uint16 {
	return s.Out
}
