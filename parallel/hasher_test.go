package parallel

import "testing"

// hasher test
func TestHasher(t *testing.T) {
	h := NewUint16Hasher(100)
	for n := uint16(0); n < 100; n++ {
		h.MustPutUint16(int(n), n)
	}
	if h.Sum()[0] != 235 {
		t.Errorf("uint16 hasher bad hash: %x", h.Sum())
	}
	h2 := NewHashHasher(100)
	for n := uint16(0); n < 100; n++ {
		h2.MustPutHash(int(n), [32]byte{byte(n)})
	}
	if h2.Sum()[0] != 77 {
		t.Errorf("hash hasher bad hash: %x", h2.Sum())
	}
}

// hasher reverse test
func TestHasherReverse(t *testing.T) {
	h := NewUint16Hasher(100)
	h.MustPutUint16(0, 0)
	for n := uint16(99); n > 0; n-- {
		h.MustPutUint16(int(n), n)
	}
	if h.Sum()[0] != 235 {
		t.Errorf("uint16 hasher bad hash: %x", h.Sum())
	}
	h2 := NewHashHasher(100)
	h2.MustPutHash(0, [32]byte{0})
	for n := uint16(99); n > 0; n-- {
		h2.MustPutHash(int(n), [32]byte{byte(n)})
	}
	if h2.Sum()[0] != 77 {
		t.Errorf("hash hasher bad hash: %x", h2.Sum())
	}
}
