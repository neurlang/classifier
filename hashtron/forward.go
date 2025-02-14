package hashtron

import "github.com/neurlang/classifier/hash"
import "sort"

// Forward calculates the hashtron output on single feaure sample (32 bit uint)
func (h Hashtron) Forward(sample uint32, negate bool) (out uint64) {
	for j := byte(0); j < h.Bits(); j++ {
		var input = uint32(sample) | (uint32(j) << 16)
		for i := 0; i < h.Len(); i++ {
			var s, max = h.Get(i)
			input = hash.Hash(input, s, max)
		}
		input &= 1
		if negate {
			input ^= 1
		}
		if input != 0 {
			out |= 1 << j
		}
	}
	return
}

type HashtronSlice []Hashtron

func (hs HashtronSlice) Forward(samples []uint32, negate int) (out []uint16) {
	if len(samples) == 0 {
		return nil
	}
	if len(samples) != len(hs) {
		panic("number of samples doesnt match number of classifiers")
	}
	if len(samples) == 1 {
		return []uint16{uint16(hs[0].Forward(samples[0], negate == 0))}
	}
	out = make([]uint16, len(hs), len(hs))
	var salts = make([]uint32, len(hs), len(hs))
	var temps = make([]uint32, len(hs), len(hs))
	var maxs = make([]uint32, len(hs), len(hs))
	var lengths_indices = make([][2]uint32, len(hs), len(hs))
	var all = len(hs)
	for i := range hs {
		if hs[i].Bits() > 1 {
			panic("multibit not supported yet on this path")
		}
		lengths_indices[i] = [2]uint32{uint32(hs[i].Len()), uint32(i)}
	}
	// Sort the slice by the first element of each sub-array
	sort.Slice(lengths_indices, func(i, j int) bool {
		return lengths_indices[i][0] > lengths_indices[j][0]
	})
	for j := range lengths_indices {
		temps[j] = samples[lengths_indices[j][1]]
	}
	for i := uint32(0); i < lengths_indices[0][0]; i++ {
		var count_before, count_after = 0, 0
		for j := range lengths_indices {
			if lengths_indices[j][0] >= i+1 {
				count_before++
			}
			if lengths_indices[j][0] > i+1 {
				count_after++
			}
		}
		for j := 0; j < count_before; j++ {
			salts[j], maxs[j] = hs[lengths_indices[j][1]].Get(int(i))
		}
		hash.HashVectorizedDistinct(samples[:count_before], temps[:count_before], salts[:count_before], maxs[:count_before])
		for j := count_after; j < count_before; j++ {
			out[lengths_indices[j][1]] = uint16(samples[j])
			all--
		}
		temps, samples = samples, temps
	}
	if all != 0 {
		panic("not all")
	}
	if negate >= 0 && negate < len(out) {
		out[negate] ^= 1
	}
	return
}
