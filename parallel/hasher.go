package parallel

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"sync"
)

type Hasher struct {
	mut  sync.Mutex
	sha  hash.Hash
	ate  int
	data [][64]byte
	done map[int]struct{}
}

func NewUint16Hasher(n int) *Hasher {
	return &Hasher{
		sha:  sha256.New(),
		data: make([][64]byte, (29+n)/30),
	}
}

func NewHashHasher(n int) *Hasher {
	return &Hasher{
		sha:  sha256.New(),
		data: make([][64]byte, (n+1)/2),
		done: make(map[int]struct{}),
	}
}

func (h *Hasher) ready() bool {
	if h.ate >= len(h.data) {
		return false
	}
	if h.done != nil {
		if _, ok := h.done[h.ate]; ok {
			delete(h.done, h.ate)
			return true
		}
	} else {
		//println(h.data[h.ate][0], h.data[h.ate][1], h.data[h.ate][62], h.data[h.ate][63])
		return (h.data[h.ate][0] == 0x7f && h.data[h.ate][1] == 0xff) &&
			(h.data[h.ate][62] == 0x7f && h.data[h.ate][63] == 0xff)
	}
	return false
}

func (h *Hasher) eat() {
	h.sha.Write(h.data[h.ate][:])
	h.ate++
}

func (h *Hasher) MustPutUint16(n int, value uint16) {
	block := n / 30
	offset := (n % 30) * 2
	position := n % 30
	
	if h.data[block][2+offset] != 0 {
		panic("uint16 write preexisting data 0")
	}
	if h.data[block][2+offset+1] != 0 {
		panic("uint16 write preexisting data 1")
	}

	h.data[block][2+offset] = byte(value)
	h.data[block][2+offset+1] = byte(value >> 8)

	h.mut.Lock()
	defer h.mut.Unlock()


	var markBytes []byte
	var pos uint
	if position < 15 {
		markBytes = h.data[block][62:64]
		pos = uint(position)
	} else {
		markBytes = h.data[block][0:2]
		pos = uint(position - 15)
	}

	currentMark := binary.BigEndian.Uint16(markBytes)
	mask := uint16(1) << pos
	if (currentMark & mask) != 0 {
		println(n, value)
		panic("duplicate write")
	}
	currentMark |= mask
	binary.BigEndian.PutUint16(markBytes, currentMark)

	for h.ready() {
		h.eat()
	}
}

func (h *Hasher) MustPutHash(n int, value [32]byte) {
	block := n >> 1
	offset := (n & 1) * 32
	
	h.mut.Lock()
	
	var markBytesOwn []byte

	if n&1 == 0 {
		markBytesOwn = h.data[block][0:2]
	} else {
		markBytesOwn = h.data[block][62:64]
	}
	ownMark := binary.BigEndian.Uint16(markBytesOwn)
	exists_opposite_hash := ownMark & uint16(1<<15) == uint16(1<<15)
	h.mut.Unlock()

	for i := 2; i < 30; i++ {
		if h.data[block][offset+i] != 0 {
			panic("hash write preexisting data")
		}
	}
	copy(h.data[block][offset:offset+32], value[:])

	h.mut.Lock()
	defer h.mut.Unlock()
	if !exists_opposite_hash {
	
		var markBytesOther []byte

		if n&1 == 0 {
			markBytesOther = h.data[block][62:64]
		} else {
			markBytesOther = h.data[block][0:2]
		}
		currentMark := binary.BigEndian.Uint16(markBytesOther)
		currentMark |= uint16(1<<15)
		binary.BigEndian.PutUint16(markBytesOther, currentMark)
	} else {
		h.done[block] = struct{}{}
	}

	for h.ready() {
		h.eat()
	}
}

func (h *Hasher) Sum() (ret [32]byte) {
	h.mut.Lock()
	for h.ate < len(h.data) {
		h.eat()
	}
	if h.ate == len(h.data) {
		copy(ret[:], h.sha.Sum(nil))
		h.ate = 0
		h.data = nil
	}
	h.mut.Unlock()
	return
}
