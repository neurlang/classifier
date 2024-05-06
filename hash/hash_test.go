package hash

import (
	"testing"
)

// performance benchmark
func BenchmarkHash(b *testing.B) {
	n := uint32(0)
	s := uint32(0)
	for i := uint32(1<<b.N); i > 1; i-- {
		n = Hash(n, s, i)
		s++
	}
}

// loop length test
func TestHash(t *testing.T) {
	const bound1 = 30
	const bound2 = 100000
	var count uint64
	for max := uint32(1); max <= 1<<bound1; max <<= 1 {
		var visited = make([]bool, max, max)
		var current uint32
		for s := uint32(0); s < bound2; s++ {
			current = Hash(current, s, max)
			if current == 0 || visited[current] {
				visited = make([]bool, max, max)
				continue
			} else {
				visited[current] = true
				count++
			}
		}
	}
	println("Tested bound1 1 <<", bound1, "bound2", bound2, "result:", count, "(higher is likely better)")
}

// sanity check fuzz
func FuzzHash(f *testing.F) {
	f.Add(uint32(0),uint32(0),uint32(0),uint32(0))
	f.Fuzz(func(t *testing.T, n, s, max, constant uint32) {
		out := Hash(n, s, max)
		if max == 0 && out != 0 {
			t.Errorf("Hard error: Hash(%d, %d, 0) == %d (max=0 should be 0)", n, s, out)
		}
		if max > 1 && out >= max {
			t.Errorf("Hard error: Hash(%d, %d, %d) == %d (output bigger or equal than max)", n, s, max, out)
		}
		// these warnings might be real, but the fuzzer is too weak to find them
		if max >= 1<<9 && out == n {
			t.Errorf("Warning: Hash(%d, %d, %d) == %d (fixed point for big max)", n, s, max, out)
		}
		if max >= 1<<9 && out == constant {
			t.Errorf("Warning: Hash(%d, %d, %d) == %d == %d (guessed output for for big max)", n, s, max, out, constant)
		}
	})
}
