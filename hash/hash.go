// Package Hash implements the fast modular hash used by the Neurlang classifier
package hash

func Hash(n uint32, s uint32, max uint32) uint32 {
	// mixing stage, mix input with salt using subtraction
	// (could also be addition)
	var m = uint32(n) - uint32(s)

	// hashing stage, use xor shift with prime coefficients
	m ^= m << 2
	m ^= m << 3
	m ^= m >> 5
	m ^= m >> 7
	m ^= m << 11
	m ^= m << 13
	m ^= m >> 17
	m ^= m << 19

	// mixing stage 2, mix input with salt using addition
	m += s

	// modular stage
	// to force output in range 0 to max-1 we could do regular modulo
	// however, the faster multiply shift trick by Daniel Lemire is used instead
	// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
	return uint32((uint64(m) * uint64(max)) >> 32)
}
