// Package Learning implements the learning stage of the Neurlang classifier
package learning

import "fmt"
import "sync"
import "math/rand"
import crypto_rand "crypto/rand"
import "time"
import "encoding/binary"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hash"

type modulo_t = uint32

func (h *HyperParameters) Training(d datasets.Splitter) {
	var sd = d.Split()

	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}

	sd = datasets.BalanceDataset(sd)

	var backup = h.InitialLimit
	var unsolved bool

	h.InitialLimit, unsolved = h.SolveN(sd[:])
	for !h.EndWhenSolved || unsolved {
		h.InitialLimit, unsolved = h.SolveN(sd[:])
	}
	h.InitialLimit = backup
}

func (h *HyperParameters) TrainingN(d datasets.SplittNer) {
	var sd = d.SplitN()

	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}

	sd = datasets.BalanceDatasetN(sd)

	var backup = h.InitialLimit
	var unsolved bool

	h.InitialLimit, unsolved = h.SolveN(sd)
	for !h.EndWhenSolved || unsolved {
		h.InitialLimit, unsolved = h.SolveN(sd)
	}
	h.InitialLimit = backup
}

func progressBar(progress, width int) string {
	progressBar := ""
	for i := 0; i < progress; i++ {
		progressBar += "="
	}
	return progressBar
}

func emptySpace(space int) string {
	emptySpace := ""
	for i := 0; i < space; i++ {
		emptySpace += " "
	}
	return emptySpace
}

func (h *HyperParameters) SolveN(d datasets.SplittedNDataset) (int, bool) {

	var bitsi byte

	var alphabet [][]uint32
	for n := range d {
		var bits uint16
		alphabet = append(alphabet, make([]uint32, 0, len(d[n])))

		for v := range d[n] {
			alphabet[n] = append(alphabet[n], v)
			bits |= uint16(v >> 16)
		}
		if n == 0 {
			for ; bits>0; bits>>=1 {
				bitsi++
			}
		}
	}

	var sols [][2]uint32
	var maxl = modulo_t(len(d[0]))
	var maxmaxl = maxl
	var max uint32 = uint32((uint64(maxl) * uint64(maxl)) / uint64(h.Factor))
	var maxmax uint32 = max
	const progressBarWidth = 40
looop:
	for max <= maxmax {
		if !h.DisableProgressBar {
			progress := progressBarWidth - int(maxl * progressBarWidth / maxmaxl)
			percent := 100 - int(maxl * 100 / maxmaxl)
			fmt.Printf("\r[%s%s] %d%% ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent)
		}
		var sol [2]uint32
		if len(d) == 2 {
			var alphabet2 = [2][]uint32{alphabet[0], alphabet[1]}
			sol = h.Reduce(max, maxl, &alphabet2)
		} else {
			sol = h.ReduceN(max, maxl, alphabet)
		}
		if sol[1] == 0 {
			if len(sols) > 0 && sols[len(sols)-1][1] > max+1 {
				max++
				continue looop
			}
			return h.InitialLimit, true
		}
		var sets = make([]map[uint32]struct{}, len(d), len(d))
		for n := range d {
			sets[n] = make(map[uint32]struct{})
		}

		for i := modulo_t(0); i < maxl; i++ {
			for n := range d {
				sets[n][hash.Hash(alphabet[n][i], sol[0], uint32(sol[1]))] = struct{}{}
			}
		}
		var biggest int
		for n := range d {
			if len(sets[n]) > biggest {
				biggest = len(sets[n])
			}
		}

		for n := range d {
			var i int
			for v := range sets[n] {
				for m := range d {
					if m == n {
						continue
					}
					if _, ok := sets[m][v]; ok {
						//panic("algorithm bad")
						continue looop
					}
				}
				alphabet[0][i] = v
				i++
			}
			for i < biggest {
				var v = rand.Uint32()
				for m := range d {
					if m == n {
						continue
					}
					if _, ok := sets[m][v]; ok {
						//panic("algorithm bad")
						continue looop
					}
				}
				alphabet[0][i] = v
				i++
			}
		}

		sols = append(sols, sol)

		maxl = modulo_t(len(sets[0]))

		if maxl < 2 {
			if len(d) == 2 {
				for val0 := range sets[0] {
					if (val0 & 1) == 1 {
						continue looop
					}
				}
				for val1 := range sets[1] {
					if (val1 & 1) == 0 {
						continue looop
					}
				}
			}
			if len(sols) < h.InitialLimit {
				if h.l != nil {
					h.l.Println("var programBits byte = ", bitsi)
					h.l.Println("var program = [][2]uint32{")
					var maxx uint32
					for i, v := range sols {
						if i == 0 {
							maxx = v[1]
						} else {
							maxx -= v[1]
						}
						h.l.Println("{", v[0], ",", maxx, "},")
						maxx = v[1]
					}
					h.l.Println("}")
					h.l.Println("// size == ", len(sols))
				}
				println("SOLUTION SIZE == ", len(sols))
			}
			if len(sols) > h.InitialLimit {
				println("SOLUTION SIZE is LIMIT ", len(sols))
				return h.InitialLimit, true
			}
			return len(sols), false
		}

		sets = nil

		var sub = h.Subtractor

		if sub >= maxl {
			sub = maxl - 1
		}

		max = uint32(uint64(max) * ((uint64(maxl-sub) * uint64(maxl-sub))) / (uint64(maxl) * uint64(maxl)))
		if max == 0 {
			max++
		}
	}
	return h.InitialLimit, true
}

// where is used to kill threads when it increases
var where byte
var mutex sync.RWMutex

func (h *HyperParameters) Reduce(max uint32, maxl modulo_t, alphabet *[2][]uint32) (off [2]uint32) {
	var out [2]uint32
	mutex.Lock()
	where++
	mutex.Unlock()
	if h.Shuffle {
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
	}
	for t := 1; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var my_where = where
			mutex.RUnlock()
		outer:
			for s := uint32(tt); true; s += uint32(h.Threads) {
				mutex.RLock()
				if my_where != where || out[0] != 0 || out[1] != 0 {
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
							if modulo_t(j) > h.Printer {
								if h.DisableProgressBar {
									println("Backtracking:", j)
								}
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
						if modulo_t(j) > h.Printer {
							if h.DisableProgressBar {
								println("Backtracking:", j)
							}
						}
						continue outer
					}
				}
				var set0 = make(map[uint32]struct{})
				var set1 = make(map[uint32]struct{})
				for j := modulo_t(0); j < maxl; j++ {

					set0[hash.Hash(alphabet[0][j], s, max)] = struct{}{}
					//println(tt,"sol=",alphabet[0][j], hash.Hash(alphabet[0][j], s, max))

				}
				//println(tt)
				for j := modulo_t(0); j < maxl; j++ {

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
					if h.DisableProgressBar {
						println("Size: ", len(set0), "Modulo:", max)
					}
					//println("{", s, ",", max, "}, // ", len(set0))
					if out[1] > uint32(max) || (out[1] == 0 && out[0] == 0) {
						out[0] = s
						out[1] = uint32(max)
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
	var deadline = h.DeadlineMs
	for out[0] == 0 && out[1] == 0 && deadline > 0 {
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
		deadline--
		mutex.RLock()
	}
	off = out
	mutex.RUnlock()
	if deadline == 0 {
		mutex.Lock()
		out[0] = ^uint32(0)
		out[1] = ^uint32(0)
		mutex.Unlock()
		if h.DisableProgressBar {
			println("Deadline")
		}
	}

	return
}


func (h *HyperParameters) ReduceN(max uint32, maxl modulo_t, alphabet [][]uint32) (off [2]uint32) {
	var out [2]uint32
	var N = uint32(len(alphabet))
	mutex.Lock()
	where++
	mutex.Unlock()
	if h.Shuffle {
		for n := range alphabet {
			rand.Shuffle(int(maxl), func(i, j int) { alphabet[n][i], alphabet[n][j] = alphabet[n][j], alphabet[n][i] })
		}
	}
	for t := 1; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var my_where = where
			mutex.RUnlock()
		outer:
			for s := uint32(tt); true; s += uint32(h.Threads) {
				mutex.RLock()
				if my_where != where || out[0] != 0 || out[1] != 0 {
					mutex.RUnlock()
					return
				} else {
					mutex.RUnlock()
				}
				var letter = make([][]uint32, N, N)
				for n := range alphabet {
					letter[n] = make([]uint32, maxl, maxl)
				}
				var set = make([]byte, max, max)

				var i uint32 = 0
				var v = alphabet[N-1][i]
				for j := uint32(0); j < N*uint32(maxl); j++ {
					if letter[j%N][i%uint32(maxl)] != 0 {
						letter[j%N][i%uint32(maxl)] = ((uint32(i) % max) + 1)
					}
					if set[(uint32(i)%max)] != byte(0) {
						if set[(uint32(i)%max)] != (byte((j)%N) + 1) {
							if modulo_t(j) > h.Printer {
								if h.DisableProgressBar {
									println("Backtracking:", j)
								}
							}
							continue outer
						}
					}

					set[(uint32(i) % max)] = byte(j%N) + 1
					//println("at", v, i)
					i = hash.Hash(v, s, max)
					v = alphabet[j%N][i%uint32(maxl)]
					//fmt.Println(letter)

				}
				i = hash.Hash(v, s, max)

				var j = uint32(N-1)
				if letter[j%N][i%uint32(maxl)] != 0 {
					letter[j%N][i%uint32(maxl)] = ((uint32(i) % max) + 1)
				}
				if set[(uint32(i)%max)] != byte(0) {
					if set[(uint32(i)%max)] != (byte((j)%N) + 1) {
						if modulo_t(j) > h.Printer {
							if h.DisableProgressBar {
								println("Backtracking:", j)
							}
						}
						continue outer
					}
				}
				var sets []map[uint32]struct{}
				for n := range alphabet {
					sets[n] = make(map[uint32]struct{})
				}
				for j := modulo_t(0); j < maxl; j++ {

					sets[0][hash.Hash(alphabet[0][j], s, max)] = struct{}{}
					//println(tt,"sol=",alphabet[0][j], hash.Hash(alphabet[0][j], s, max))

				}
				//println(tt)
				for n := uint32(1); n < N; n++ {
					for j := modulo_t(0); j < maxl; j++ {

						var val = hash.Hash(alphabet[1][j], s, max)
						sets[n][val] = struct{}{}

						for m := uint32(0); m < N; m++ {
							if m == n {
								continue
							}
							if _, ok := sets[m][val]; ok {
								//panic("algorithm bad")
								continue outer
							}
						}


						//println(tt,"sol=",alphabet[1][j], hash.Hash(alphabet[1][j], s, max))

					}
				}
				//panic(tt)
/*
				for n := range sets {
					if len(sets[0]) != len(sets[n]) {
						continue outer
					}
				}
*/
				mutex.Lock()
				if h.DisableProgressBar {
					println("Size: ", len(sets[0]), "Modulo:", max)
				}
				//println("{", s, ",", max, "}, // ", len(set0))
				if out[1] > uint32(max) || (out[1] == 0 && out[0] == 0) {
					out[0] = s
					out[1] = uint32(max)
				}
				mutex.Unlock()
				return
			}
		}(byte(t))
	}
	mutex.RLock()
	var deadline = h.DeadlineMs
	for out[0] == 0 && out[1] == 0 && deadline > 0 {
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
		deadline--
		mutex.RLock()
	}
	off = out
	mutex.RUnlock()
	if deadline == 0 {
		mutex.Lock()
		out[0] = ^uint32(0)
		out[1] = ^uint32(0)
		mutex.Unlock()
		if h.DisableProgressBar {
			println("Deadline")
		}
	}

	return
}
