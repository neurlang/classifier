package avx

func scatterGatherNotVectorized(outs []uint32, buf []uint32, size *int, wh uint32) bool {
	const subwords = 16
	const twobitmask = 3
	for i := range outs { // parallel - in any order 
		v := outs[i]
		w0 := v / subwords
		w1 := (v % subwords) << 1
		loaded := (buf[w0] >> w1) & twobitmask
		if loaded == uint32(2 - wh) {
			return true
		}
		if loaded == 0 {
			*size++
		}
		buf[w0] |= uint32(1 + wh) << w1
	}
	return false
}
