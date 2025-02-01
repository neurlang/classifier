package avx

func scatterGatherNotVectorized(outs []uint32, buf []uint32, size *int, wh uint32) bool {
	for i := range outs { // parallel - in any order 
		v := outs[i]
		if buf[v] == uint32(2 - wh) {
			return true
		}
		if buf[v] == 0 {
			*size++
		}
		buf[v] |= uint32(1 + wh)
	}
	return false
}
