package parallel

import "testing"

// hasher test
func TestHasher(t *testing.T) {
	h := NewUint16Hasher(100)
	for n := uint16(0); n < 100; n++ {
		h.MustPutUint16(int(n), n)
	}
	if h.Sum()[0] != 206 {
		t.Errorf("uint16 hasher bad hash: %x", h.Sum())
	}
	h2 := NewHashHasher(100)
	for n := uint16(0); n < 100; n++ {
		h2.MustPutHash(int(n), [32]byte{})
	}
	if h2.Sum()[0] != 139 {
		t.Errorf("hash hasher bad hash: %x", h2.Sum())
	}
}
