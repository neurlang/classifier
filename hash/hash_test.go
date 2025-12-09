package hash

import (
	"testing"
)

// performance benchmark
func BenchmarkHash(b *testing.B) {
	n := uint32(0)
	s := uint32(0)
	for i := uint32(1 << b.N); i > 1; i-- {
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
	f.Add(uint32(0), uint32(0), uint32(0), uint32(0))
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

// TestHashVectorized verifies that HashVectorized produces the same results as multiple Hash() calls
func TestHashVectorized(t *testing.T) {
	testCases := []struct {
		name string
		size int
	}{
		{"single", 1},
		{"small", 8},
		{"medium", 16},
		{"large", 64},
		{"odd", 17},
		{"prime", 31},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			size := tc.size
			n := make([]uint32, size)
			s := make([]uint32, size)
			out := make([]uint32, size)
			expected := make([]uint32, size)

			// Fill with test data
			for i := 0; i < size; i++ {
				n[i] = uint32(i*123 + 456)
				s[i] = uint32(i*789 + 101112)
			}

			max := uint32(1000000)

			// Compute expected results using Hash()
			for i := 0; i < size; i++ {
				expected[i] = Hash(n[i], s[i], max)
			}

			// Compute using HashVectorized
			HashVectorized(out, n, s, max)

			// Compare results
			for i := 0; i < size; i++ {
				if out[i] != expected[i] {
					t.Errorf("HashVectorized mismatch at index %d: got %d, want %d (n=%d, s=%d, max=%d)",
						i, out[i], expected[i], n[i], s[i], max)
				}
			}
		})
	}
}

// TestHashVectorizedDistinct verifies that HashVectorizedDistinct produces the same results as multiple Hash() calls
func TestHashVectorizedDistinct(t *testing.T) {
	testCases := []struct {
		name string
		size int
	}{
		{"single", 1},
		{"small", 8},
		{"medium", 16},
		{"large", 64},
		{"odd", 17},
		{"prime", 31},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			size := tc.size
			n := make([]uint32, size)
			s := make([]uint32, size)
			max := make([]uint32, size)
			out := make([]uint32, size)
			expected := make([]uint32, size)

			// Fill with test data - each element has different max
			for i := 0; i < size; i++ {
				n[i] = uint32(i*123 + 456)
				s[i] = uint32(i*789 + 101112)
				max[i] = uint32((i+1) * 50000)
			}

			// Compute expected results using Hash()
			for i := 0; i < size; i++ {
				expected[i] = Hash(n[i], s[i], max[i])
			}

			// Compute using HashVectorizedDistinct
			HashVectorizedDistinct(out, n, s, max)

			// Compare results
			for i := 0; i < size; i++ {
				if out[i] != expected[i] {
					t.Errorf("HashVectorizedDistinct mismatch at index %d: got %d, want %d (n=%d, s=%d, max=%d)",
						i, out[i], expected[i], n[i], s[i], max[i])
				}
			}
		})
	}
}

// TestHashVectorizedEdgeCases tests edge cases for vectorized hash functions
func TestHashVectorizedEdgeCases(t *testing.T) {
	t.Run("zero_max", func(t *testing.T) {
		n := []uint32{1, 2, 3, 4}
		s := []uint32{5, 6, 7, 8}
		out := make([]uint32, 4)
		expected := make([]uint32, 4)

		max := uint32(0)
		for i := range n {
			expected[i] = Hash(n[i], s[i], max)
		}

		HashVectorized(out, n, s, max)

		for i := range out {
			if out[i] != expected[i] {
				t.Errorf("zero_max: index %d: got %d, want %d", i, out[i], expected[i])
			}
		}
	})

	t.Run("max_uint32", func(t *testing.T) {
		n := []uint32{1, 2, 3, 4}
		s := []uint32{5, 6, 7, 8}
		out := make([]uint32, 4)
		expected := make([]uint32, 4)

		max := uint32(0xFFFFFFFF)
		for i := range n {
			expected[i] = Hash(n[i], s[i], max)
		}

		HashVectorized(out, n, s, max)

		for i := range out {
			if out[i] != expected[i] {
				t.Errorf("max_uint32: index %d: got %d, want %d", i, out[i], expected[i])
			}
		}
	})

	t.Run("distinct_zero_max", func(t *testing.T) {
		n := []uint32{1, 2, 3, 4}
		s := []uint32{5, 6, 7, 8}
		max := []uint32{0, 100, 0, 200}
		out := make([]uint32, 4)
		expected := make([]uint32, 4)

		for i := range n {
			expected[i] = Hash(n[i], s[i], max[i])
		}

		HashVectorizedDistinct(out, n, s, max)

		for i := range out {
			if out[i] != expected[i] {
				t.Errorf("distinct_zero_max: index %d: got %d, want %d", i, out[i], expected[i])
			}
		}
	})
}
