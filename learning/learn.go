// Package Learning implements the learning stage of the Neurlang classifier
package learning

import "sync"
import "math/rand"
import crypto_rand "crypto/rand"
import "time"
import "encoding/binary"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hash"

func (h *HyperParameters) Training(d datasets.Dataset) {
	var sd = datasets.SplitDataset(d)
	sd = datasets.BalanceDataset(sd)

	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}

	for {
		h.InitialLimit = h.Solve(sd)
	}
}

func (h *HyperParameters) Solve(d datasets.SplittedDataset) int {

	var alphabet = [2][]uint32{
		make([]uint32, 0, len(d[0])),
		make([]uint32, 0, len(d[1])),
	}

	for v := range d[0] {
		alphabet[0] = append(alphabet[0], v)
	}
	for v := range d[1] {
		alphabet[1] = append(alphabet[1], v)
	}

	var sols = [][2]uint32{}
	var max uint32 = h.InitialModulo
	var maxl = uint16(len(d[0]))
	var maxmax uint32 = h.InitialModulo

looop:
	for max <= maxmax {
		var sol = h.Reduce(max, maxl, &alphabet)
		if sol[1] == 0 {
			if len(sols) > 0 && sols[len(sols)-1][1] > max + 1 {
				max ++
				continue looop
			}
			return h.InitialLimit
		}
		var set0i = make(map[uint32]struct{})
		var set1i = make(map[uint32]struct{})
		for i := uint16(0); i < maxl; i++ {
			set0i[hash.Hash(alphabet[0][i], sol[0], uint32(sol[1]))] = struct{}{}
			set1i[hash.Hash(alphabet[1][i], sol[0], uint32(sol[1]))] = struct{}{}
		}

		{
			var i int
			for v := range set0i {
				if _, ok := set1i[v]; ok {
					//panic("algorithm bad")
					continue looop
				}
				alphabet[0][i] = v
				i++
			}
			i = 0
			for v := range set1i {
				if _, ok := set0i[v]; ok {
					//panic("algorithm bad")
					continue looop
				}
				alphabet[1][i] = v
				i++
			}
		}

		maxl = uint16(len(set0i))


		sols = append(sols, sol)

		if maxl < 2 {
			for val0 := range set0i {
				if (val0 & 1) == 1 {
					continue looop
				}
			}
			for val1 := range set1i {
				if (val1 & 1) == 0 {
					continue looop
				}
			}

			if len(sols) < h.InitialLimit {
				if h.l != nil {
					h.l.Println("var program = [][2]uint32{")
					for _, v := range sols {
						h.l.Println("{", v[0], ",", v[1], "},")
					}
					h.l.Println("}")
					h.l.Println("// size == ", len(sols))
				}
				println("SOLUTION SIZE == ", len(sols))
			}
			if len(sols) > h.InitialLimit {
				println("SOLUTION SIZE is LIMIT ", len(sols))
				return h.InitialLimit
			}
			return len(sols)
		}

		set0i = nil
		set1i = nil

		max *= h.Numerator
		max /= h.Denominator
		max -= h.Subtractor
	}
	return h.InitialLimit
}

// where is used to kill threads when it increases
var where byte
var mutex sync.RWMutex

func (h *HyperParameters) Reduce(max uint32, maxl uint16, alphabet *[2][]uint32) (off [2]uint32) {
	var out [256][2]uint32
	mutex.Lock()
	where++
	out[where][0] = 0
	out[where][1] = 0
	mutex.Unlock()
	if h.Shuffle {
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
	}
	for t := 1; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var where = where
			mutex.RUnlock()
		outer:
			for s := uint32(tt); true; s += uint32(h.Threads) {
				mutex.RLock()
				if out[where][0] != 0 || out[where][1] != 0 {
					mutex.RUnlock()
					return
				} else {
					mutex.RUnlock()
				}
				var letter = [2][]uint32{make([]uint32, maxl, maxl), make([]uint32, maxl, maxl)}
				var set = make([]byte, max, max)

				var i uint32 = 0
				var v = alphabet[1][i]
				for j := uint32(0); j < 2*uint32(maxl); j++ {
					if letter[j&1][i%uint32(maxl)] != 0 {
						letter[j&1][i%uint32(maxl)] = ((uint32(i) % max) + 1)
					}
					if set[(uint32(i)%max)] != byte(0) {
						if set[(uint32(i)%max)] == (byte((j^1)&1) + 1) {
							if uint16(j) > h.Printer {
								println("Backtracking:", j)
							}
							continue outer
						}
					}

					set[(uint32(i) % max)] = byte(j&1) + 1
					//println("at", v, i)
					i = hash.Hash(v, s, max)
					v = alphabet[j&1][i%uint32(maxl)]
					//fmt.Println(letter)

				}
				i = hash.Hash(v, s, max)
				var j = uint32(1)
				if letter[j&1][i%uint32(maxl)] != 0 {
					letter[j&1][i%uint32(maxl)] = ((uint32(i) % max) + 1)
				}
				if set[(uint32(i)%max)] != byte(0) {
					if set[(uint32(i)%max)] == (byte((j^1)&1) + 1) {
						if uint16(j) > h.Printer {
							println("Backtracking:", j)
						}
						continue outer
					}
				}
				var set0 = make(map[uint32]struct{})
				var set1 = make(map[uint32]struct{})
				for j := uint16(0); j < maxl; j++ {
					set0[hash.Hash(alphabet[0][j], s, max)] = struct{}{}
					//println(tt,"sol=",alphabet[0][j], hash.Hash(alphabet[0][j], s, max))

				}
				//println(tt)
				for j := uint16(0); j < maxl; j++ {
					var val = hash.Hash(alphabet[1][j], s, max)
					set1[val] = struct{}{}
					if _, ok := set0[val]; ok {
						//panic("algorithm bad")
						continue outer
					}
					//println(tt,"sol=",alphabet[1][j], hash.Hash(alphabet[1][j], s, max))

				}
				//panic(tt)
				if len(set0) == len(set1) {
					mutex.Lock()
					println("Size: ", len(set0), "Modulo:", max)
					//println("{", s, ",", max, "}, // ", len(set0))
					if out[where][1] > uint32(max) || (out[where][1] == 0 && out[where][0] == 0) {
						out[where][0] = s
						out[where][1] = uint32(max)
					}
					mutex.Unlock()
					return
				} else {
					continue outer
				}
			}
		}(byte(t))
	}
	mutex.RLock()
	var where = where
	var deadline = h.DeadlineMs
	for out[where][0] == 0 && out[where][1] == 0 && deadline > 0 {
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
		deadline--
		mutex.RLock()
	}
	off = out[where]
	mutex.RUnlock()
	if deadline == 0 {
		mutex.Lock()
		out[where][0] = ^uint32(0)
		out[where][1] = ^uint32(0)
		mutex.Unlock()
		println("Deadline")
	}

	return
}
