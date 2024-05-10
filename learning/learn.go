// Package Learning implements the learning stage of the Neurlang classifier
package learning

import "fmt"
import "sync"
import "math/rand"
import crypto_rand "crypto/rand"
import "time"
import "encoding/binary"
import "sort"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/hashtron"

type modulo_t = uint32

func (h *HyperParameters) Training(d datasets.Splitter) (*hashtron.Hashtron, error) {

	if h.EOL == nil || len(h.EOL) == 0 {
		h.EOL = []byte{';', ' '}
	}

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
	var result *hashtron.Hashtron

	h.InitialLimit, result = h.Solve(sd)
	for !h.EndWhenSolved || result == nil {
		h.InitialLimit, result = h.Solve(sd)
	}
	h.InitialLimit = backup

	return result, nil
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

func (h *HyperParameters) Solve(d datasets.SplittedDataset) (int, *hashtron.Hashtron) {

	if len(d[1]) == 0 && len(d[0]) == 0 {
		tron, err := hashtron.New(nil, 0)
		if err != nil {
			return h.InitialLimit, nil
		}
		return 1, tron
	}

	var bits uint16
	var alphabet [][]uint32
	for n := range d {

		alphabet = append(alphabet, make([]uint32, 0, len(d[n])))

		for v := range d[n] {
			alphabet[n] = append(alphabet[n], v)
			if n == 0 && bits < uint16(v>>16) {
				bits = uint16(v >> 16)
			}

		}
	}
	if bits >= 64 {
		bits = 0
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
			if maxmaxl > 0 {
				progress := progressBarWidth - int(maxl*progressBarWidth/maxmaxl)
				percent := 100 - int(maxl*100/maxmaxl)
				fmt.Printf("\r[%s%s] %d%% ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent)
			}
		}
		var alphabet2 = [2][]uint32{alphabet[0], alphabet[1]}
		var sol [2]uint32
		var size int
		if max == 1 && maxl == 1 {
			sol = h.Reduce1(&alphabet2)
			size = 1
		} else {
			sol, size = h.Reduce(max, maxl, &alphabet2)
		}
		if sol[1] == 0 {
			if len(sols) > 0 && sols[len(sols)-1][1] > max+1 {
				max++
				continue looop
			}
			return h.InitialLimit, nil
		}
		var set = make(map[uint32]struct{})

		for n := range d {
			dst := alphabet[n][0:0:maxl]
			for i := modulo_t(0); i < maxl; i++ {
				hsh := hash.Hash(alphabet[n][i], sol[0], uint32(sol[1]))
				if _, ok := set[hsh]; ok {
					continue
				}
				dst = append(dst, hsh)
				set[hsh] = struct{}{}
			}
			alphabet[n] = dst
		}
		set = nil

		sols = append(sols, sol)

		maxl = modulo_t(size)
		alphabet[0] = alphabet[0][0:maxl]
		alphabet[1] = alphabet[1][0:maxl]
		if maxl < 2 {
			if len(d) == 2 {
				if alphabet[0][0]&1 == 1 {
					continue looop
				}
				if alphabet[1][0]&1 == 0 {
					continue looop
				}
			}

			if len(sols) >= h.InitialLimit {
				println("SOLUTION SIZE", len(sols), "is below LIMIT ", h.InitialLimit)
				return h.InitialLimit, nil
			}
			var v1decrease uint32
			if len(sols) > 0 {
				v1decrease = sols[0][1] * 2
			}
			for i := range sols {
				sols[i][1], v1decrease = v1decrease-sols[i][1], sols[i][1]
			}
			tron, err := hashtron.New(sols, byte(bits)+1)
			if err != nil {
				println("Error creating hashtron:", err.Error())
				return h.InitialLimit, nil
			}
			if h.l != nil {
				buf, err := tron.BytesBuffer(h.Name, h.EOL...)
				if err != nil {
					println("Hashtron serialization problem:", err.Error())
				} else {
					h.l.Println(buf)
					println("SOLUTION saved! SIZE == ", len(sols))
				}
			} else {
				println("SOLUTION! SIZE == ", len(sols))
			}
			return len(sols), tron
		}

		var sub = h.Subtractor

		if sub >= maxl {
			sub = maxl - 1
		}

		max = uint32(uint64(max) * (uint64(maxl-sub) * uint64(maxl-sub)) / (uint64(maxl) * uint64(maxl)))
		if max <= 1 {
			max = 1
		}
	}
	return h.InitialLimit, nil
}

func (h *HyperParameters) Reduce1(alphabet *[2][]uint32) (off [2]uint32) {
	var out [2]uint32
	mutex.Lock()
	where++
	mutex.Unlock()
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
				if hash.Hash(alphabet[0][0], s, 2)&1 != 0 {
					continue outer
				}
				if hash.Hash(alphabet[1][0], s, 2)&1 != 1 {
					continue outer
				}
				mutex.Lock()
				if h.DisableProgressBar {
					println("Size: ", "1", "Modulo:", "2")
				}
				//println("{", s, ",", max, "}, // ", len(set0))
				if out[1] > 2 || (out[1] == 0 && out[0] == 0) {
					out[0] = s
					out[1] = 2
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

// where is used to kill threads when it increases
var where byte
var mutex sync.RWMutex

func (h *HyperParameters) Reduce(max uint32, maxl modulo_t, alphabet *[2][]uint32) (off [2]uint32, sizeoff int) {
	var out [2]uint32
	var sizeglobal int
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
				var set = make([]byte, (max+3)/4, (max+3)/4)

				{
					var i uint32 = 0
					var v = alphabet[1][i]
					for j := uint32(0); j < 2*uint32(maxl); j++ {

						//println("at", v, i)
						i = hash.Hash(v, s, max)
						v = alphabet[j&1][i%uint32(maxl)]
						//fmt.Println(letter)

						if letter[j&1][i%uint32(maxl)] != 0 {
							letter[j&1][i%uint32(maxl)] = ((uint32(i) % max) + 1)
						}
						imodmax := (uint32(i) % max)
						if (set[imodmax>>2]>>((imodmax&3)<<1))&3 != byte(0) {
							if (set[imodmax>>2]>>((imodmax&3)<<1))&3 == (byte((j^1)&1) + 1) {
								if modulo_t(j) > h.Printer {
									if h.DisableProgressBar {
										println("Backtracking:", j)
									}
								}
								continue outer
							}
						}
						set[imodmax>>2] |= (byte(j&1) + 1) << ((imodmax & 3) << 1)
					}
				}

				set = nil

				for i := range letter[0] {
					letter[0][i] = hash.Hash(alphabet[0][i], s, max)
				}
				for i := range letter[1] {
					letter[1][i] = hash.Hash(alphabet[1][i], s, max)
				}

				sort.Slice(letter[0], func(i, j int) bool { return letter[0][i] < letter[0][j] })
				sort.Slice(letter[1], func(i, j int) bool { return letter[1][i] < letter[1][j] })
				q, size := 0, 0
				{
					// Indices for the two slices
					i, j := 0, 0
					for i < len(letter[0]) && j < len(letter[1]) {
						if letter[0][i] == letter[1][j] {
							//panic("algorithm bad")
							continue outer
						} else if letter[0][i] < letter[1][j] {
							i++
							if i < len(letter[0]) && letter[0][i] == letter[0][i-1] {
								q--
							} else {
								size++
							}
						} else {
							j++
							if j < len(letter[1]) && letter[1][j] == letter[1][j-1] {
								q++
							}
						}
					}
					for i < len(letter[0]) {
						i++
						if i < len(letter[0]) && letter[0][i] == letter[0][i-1] {
							q--
						} else {
							size++
						}
					}
					for j < len(letter[1]) {
						j++
						if j < len(letter[1]) && letter[1][j] == letter[1][j-1] {
							q++
						}
					}
				}

				letter[0] = nil
				letter[1] = nil

				if q == 0 {
					mutex.Lock()
					if h.DisableProgressBar {
						println("Size: ", size, "Modulo:", max)
					}
					//println("{", s, ",", max, "}, // ", len(set0))
					if out[1] > uint32(max) || (out[1] == 0 && out[0] == 0) {
						out[0] = s
						out[1] = uint32(max)
						sizeglobal = size
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
	sizeoff = sizeglobal
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
