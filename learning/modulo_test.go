package learning

import (
	"testing"
)

// we dont care about fast modulo being slightly wrong
/*
func TestEquality(t *testing.T) {
	var mod = uint32(500000)
	recip := real_modulo_recip(mod)
	var wrong uint64
	var first uint32
	var last uint32
	for i := uint32(0); i < uint32(1000000); i++ {
		if real_modulo(i, recip, mod) != i % mod {
			wrong++
			last = i
			if first == 0 {
				first = i
			}
		}
	}
	if wrong > 0 {
		t.Errorf("Wrong result for %d .. %d, and total %d times", first, last, wrong)
	}
}
*/

func BenchmarkModOperator(b *testing.B) {
	var out uint32
	for i := uint32(0); i < uint32(b.N); i++ {
		out ^= i % 10000
	}
	println(uint32(b.N), out)
}
func BenchmarkTwiceModOperator(b *testing.B) {
	var out uint32
	var mod = uint32(b.N)
	for i := uint32(0); i < 2*uint32(b.N); i++ {
		out ^= i % mod
	}
	println(uint32(b.N), out)
}

func BenchmarkRealModulo(b *testing.B) {
	recip := real_modulo_recip(10000)
	var out uint32
	for i := uint32(0); i < uint32(b.N); i++ {
		out ^= real_modulo(i, recip, 10000)
	}
	println(uint32(b.N), out)
}

func BenchmarkTwiceIfModulo(b *testing.B) {
	var out uint32
	var mod = uint32(b.N)
	for i := uint32(0); i < 2*uint32(b.N); i++ {
		if i >= mod {
			out ^= i - mod
		} else {
			out ^= i
		}
	}
	println(uint32(b.N), out)
}
