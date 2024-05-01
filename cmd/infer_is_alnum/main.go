package main

import "github.com/neurlang/classifier/inference"

type Program [][2]uint32

func (p Program) Len() int {
	return len(p)
}
func (p Program) Get(n int) (uint32, uint32) {
	return (p)[n][0], (p)[n][1]
}
func (p Program) Bits() byte {
	return 0 // unused, return 0
}

func main() {
	var model = Program(program)

	for i := ' '; i < '~'; i++ {

		println(i, string([]rune{rune(i)}), inference.BoolInfer(uint32(i), model))

	}
}
