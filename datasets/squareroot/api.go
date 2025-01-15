package squareroot

import "math"

type Sample uint32

func (s *Sample) Feature(n int) uint32 {
	return uint32(*s)
}

func (s *Sample) Parity() uint16 {
	// don't balance
	return 4
	//return uint16(*s) ^ uint16(*s) << 3
}

func (s *Sample) Output() uint16 {
	return uint16(math.Sqrt(float64(*s)))
}

const SmallClasses = 4
const MediumClasses = 5
const BigClasses = 6
const HugeClasses = 7

func Small() (ret []Sample) {
	for i := uint16(0); i < 1<<8; i++ {
		ret = append(ret, Sample(i))
	}
	return
}

func Medium() (ret []Sample) {
	for i := uint32(0); i < 1<<10; i++ {
		ret = append(ret, Sample(i))
	}
	return
}

func Big() (ret []Sample) {
	for i := uint32(0); i < 1<<12; i++ {
		ret = append(ret, Sample(i))
	}
	return
}

func Huge() (ret []Sample) {
	for i := uint64(0); i < 1<<14; i++ {
		ret = append(ret, Sample(i))
	}
	return
}
