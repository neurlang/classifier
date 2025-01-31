// Package AVX implements the learning stage of the Neurlang classifier on AVX
package avx

//import "math/bits"
import "fmt"
import "sync"
import "math/rand"
import crypto_rand "crypto/rand"
import "time"
import "encoding/binary"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/hashtron"
import "github.com/klauspost/cpuid/v2"

type modulo_t = uint32

// Training trains a single hashtron on a dataset d. It outputs the trained hashtron if successful, or an error.
func (h *HyperParameters) Training(d datasets.Splitter) (*hashtron.Hashtron, error) {

	if h.AvxSkip == 0 { // fix wrong setup
		h.AvxSkip = 1
	}

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

// Solve directly solves a single hashtron on a splitted dataset d. It outputs the size of solution
// and the trained hashtron if successful. Most callers should use Training instead.
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
	var center uint32

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
		if maxl == 1 {
			sol = h.Reduce1(&alphabet2)
		} else if maxl == 2 {
			sol = h.Reduce2(&alphabet2)
		} else {
			sol = h.reduce(center, max, maxl, &alphabet2)
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
		var correct = len(alphabet[0])+len(alphabet[1]) < 10
		if correct {
		outer:
			for n := uint32(0); n < 2; n++ {
				for _, v := range alphabet[n] {
					if v&1 != n {
						correct = false
						break outer
					}
				}
			}
		}
		for n := 0; n < 2; n++ {
			for len(alphabet[n]) < len(alphabet[n^1]) {
				var w = uint32(uint16(rand.Uint32()))
				if _, ok := set[w]; ok {
					continue
				}
				alphabet[n] = append(alphabet[n], w)
				set[w] = struct{}{}
			}
		}
		set = nil

		sols = append(sols, sol)
		center = sol[0]
		maxl = modulo_t(len(alphabet[0]))
		maxmax = max

		if correct || maxl < 2 {
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
		if max <= maxl {
			max = maxl + 1
		}
	}
	return h.InitialLimit, nil
}

// we do this only once for a faster modulo
func real_modulo_recip(y uint32) uint32 {
	return uint32((uint64(1 << 32)) / (uint64(y)))
}

// X % Y = (BITAND(CEILING(X*256/Y),255)*Y)>>8
// manually inlined in reduce
func real_modulo(x, recip, y uint32) uint32 {
	return uint32((uint64(uint32((x+1)*recip)) * uint64(y)) >> 32)
}

// where is used to kill threads when it increases
var where byte
var mutex sync.RWMutex

func allocate(sets [][]byte, maxx uint32) {
	maxx = (maxx+3)/4 + 7 // size plus word padding
	data := make([]byte, len(sets)*int(maxx), len(sets)*int(maxx))

	// sets[0] has to be lowest pointer
	for i := range sets {
		sets[i] = data[0:maxx]
		data = data[maxx:]
	}
	//all other sets pointers will be sorted and near each other
}

func prealpha(v []uint32, alphabet *[2][]uint32, i []uint32) {
	for ii := range v {
		v[ii] = alphabet[0][i[ii]]
	}
}

func salt(salts []uint32, center, s, lanes uint32) {
	for ii := range salts {
		salts[ii] = center ^ (s + uint32(ii)*lanes)
	}
}
func postalpha(v []uint32, alphabet *[2][]uint32, i []uint32, maxl_recip, j uint32, maxl modulo_t) {
	for ii := range v {
		v[ii] = alphabet[j&1][uint32((uint64(((i[ii]+1)*maxl_recip))*uint64(maxl))>>32)]
	}
}
func imod(v []uint32, alphabet *[2][]uint32, sets [][]byte, imodmax []uint32, j uint32, callback func(ii int)) {
	for ii := range v {
		//if (sets[ii][imodmax[ii]>>2]>>((imodmax[ii]&3)<<1))&3 != byte(0) {
		if (sets[ii][imodmax[ii]>>2]>>((imodmax[ii]&3)<<1))&3 == (byte((j^1)&1) + 1) {
			//if modulo_t(j) > h.Printer {
			//	if h.DisableProgressBar {
			//		println("Backtracking:", j)
			//	}
			//}
			for i := range sets[ii] {
				sets[ii][i] = 0
			}
			imodmax[ii] = 0
			v[ii] = alphabet[0][imodmax[ii]]
			//continue outer
			//callback(ii)
		}
		//}
	}
}

var setoralloc func(sets [][]byte, imodmax []uint32, j uint32)

func setorallocAVX512Vectorized(sets **uint8, imodmax *uint32, j uint32, len uint32)

var lCPI0_0 = [16]uint32{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}

func setorallocNotVectorized(sets [][]byte, imodmax []uint32, j uint32) {
	for ii := range imodmax {
		sets[ii][imodmax[ii]>>2] |= (byte(j&1) + 1) << ((imodmax[ii] & 3) << 1)
	}
}

func init() {
	// Check if the CPU supports AVX512
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		setoralloc = func(sets [][]byte, imodmax []uint32, j uint32) {
			//expp := make([]byte, len(imodmax), len(imodmax))
			//for ii := range imodmax {
			//	expp[ii] = sets[ii][imodmax[ii]>>2]
			//}
			setp := make([]*byte, len(sets), len(sets))
			for i := range sets {
				setp[i] = &sets[i][0]
			}
			//fmt.Println(setp)
			//setorallocNotVectorized(sets, imodmax, j)
			setorallocAVX512Vectorized(&setp[0], &imodmax[0], j, uint32(len(imodmax)))
			//for ii := range imodmax {
			//	if sets[ii][imodmax[ii]>>2] != expp[ii] | (byte(j&1) + 1) << ((imodmax[ii] & 3) << 1) {
			//		println("self checking failed", ii, expp[ii], sets[ii][imodmax[ii]>>2], sets[ii][0], sets[ii][1], sets[ii][2], sets[ii][3], )
			//	}
			//}
		}
	} else {
		setoralloc = setorallocNotVectorized
	}
}

func (h *HyperParameters) reduce(center, maxx uint32, maxl modulo_t, alphabet *[2][]uint32) (off [2]uint32) {
	var out [2]uint32
	mutex.Lock()
	where++
	mutex.Unlock()
	if h.Shuffle {
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
	}
	for t := 0; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var my_where = where
			mutex.RUnlock()
			for s := uint32(tt); true; s += uint32(h.Threads) * uint32(h.AvxLanes) * uint32(h.AvxSkip) {
				mutex.RLock()
				if my_where != where || out[0] != 0 || out[1] != 0 {
					mutex.RUnlock()
					return
				} else {
					mutex.RUnlock()
				}
				if maxl > 4 {
					var sets = make([][]byte, h.AvxLanes, h.AvxLanes)
					allocate(sets, maxx)
					maxl_recip := real_modulo_recip(maxl)
					var i = make([]uint32, h.AvxLanes, h.AvxLanes)
					var v = make([]uint32, h.AvxLanes, h.AvxLanes)
					prealpha(v, alphabet, i)
					var salts = make([]uint32, h.AvxLanes, h.AvxLanes)
					salt(salts, center, s, uint32(h.Threads))
					//fmt.Println(tt, s, salts)
					for j := uint32(0); j < 2*uint32(maxl); j++ {

						//println("at", v, i)

						hash.HashVectorized(i, v, salts, maxx)
						postalpha(v, alphabet, i, maxl_recip, j, maxl)
						//fmt.Println(letter)

						// imodmax = i % maxx, but i is at worst 2*maxx-1
						//imodmax := i
						imod(v, alphabet, sets, i, j, func(ii int) {
							salts[ii] ^= center
							salts[ii] += uint32(h.Threads) * uint32(h.AvxLanes)
							salts[ii] ^= center
						})

						setoralloc(sets, i, j)

					}
					// exit if other thread won
					mutex.RLock()
					if my_where != where || out[0] != 0 || out[1] != 0 {
						mutex.RUnlock()
						return
					} else {
						mutex.RUnlock()
					}
				}

				var salts = make([]uint32, h.AvxLanes, h.AvxLanes)
				salt(salts, center, s, uint32(h.Threads))
				// cubic verify
				//var int0, int1 uint64
			lane:
				for q := uint32(0); q < h.AvxLanes; q++ {
					s := salts[q]
					for i := uint32(0); i < maxl; i++ {
						var v = hash.Hash(alphabet[0][i], center^s, maxx)
						//int0 |= 1 << v
						for j := uint32(0); j < maxl; j++ {
							var w = hash.Hash(alphabet[1][j], center^s, maxx)
							if v == w {
								continue lane
							}
							//int1 |= 1 << v
						}
					}
					// enforce match for small solutions
					//int0 = uint64(bits.OnesCount64(int0))
					//int1 = uint64(bits.OnesCount64(int1))
					//if maxx < 64 && int0 != int1 {
					//	continue outer
					//}
					mutex.RLock()
					if my_where != where || out[0] != 0 || out[1] != 0 {
						mutex.RUnlock()
						return
					} else {
						mutex.RUnlock()
					}
					mutex.Lock()
					if h.DisableProgressBar {
						println("Size: ", maxl, "Modulo:", maxx)
					}
					//println("{", s, ",", maxx, "}, // ", len(set0))
					if out[1] > uint32(maxx) || (out[1] == 0 && out[0] == 0) {
						out[0] = center ^ s
						out[1] = uint32(maxx)
					}
					mutex.Unlock()
					return
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
		where++
		mutex.Unlock()
		if h.DisableProgressBar {
			println("Deadline")
		}
	}
	return
}
