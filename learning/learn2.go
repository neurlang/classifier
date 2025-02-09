package learning

import crypto_rand "crypto/rand"
import normal_rand "math/rand"
import "sync"
import "github.com/neurlang/classifier/parallel"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/learning/avx"
import "encoding/binary"
import "fmt"
import "time"

//import "slices"

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

func timeTrack(start time.Time) {
	elapsed := time.Since(start)
	fmt.Printf("time: %s\n", elapsed)
}

// Reducing could mutate the alphabet, and it runs only till untilMaxl if nonzero, starting from initMaxx if nonzero
func (h *HyperParameters) Reducing(alphabet [2][]uint32, untilMaxl, initMaxx uint32) [][2]uint32 {
	//defer timeTrack(time.Now())
	if len(alphabet[0])+len(alphabet[1]) == 0 {
		// garbage in, garbage out
		return nil
	}
	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			normal_rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}
	// add random value to empty set not present in the other set
	{
		for i := 0; i < 2; i++ {
			if len(alphabet[i]) == 0 {
				// roll a random number
				var rand = normal_rand.Uint32()
				var exists bool
				// The random number can't be in the other set
				for _, v := range alphabet[1-i] {
					if v == rand {
						exists = true
						break
					}
				}
				if exists {
					// Find the first missing number
					rand = uint32(len(alphabet[1-i]))
					for j := range alphabet[1-i] {
						for alphabet[1-i][j] < uint32(j) && alphabet[1-i][j] != alphabet[1-i][alphabet[1-i][j]] {
							aa := &alphabet[1-i][j]
							bb := &alphabet[1-i][alphabet[1-i][j]]
							*aa, *bb = *bb, *aa
						}
					}
					// now each small number was swapped to it's value's place
					for j, v := range alphabet[1-i] {
						if v != uint32(j) {
							rand = uint32(j)
							break
						}
					}
				}
				// add it to current alphabet
				alphabet[i] = append(alphabet[i], rand)
			}
		}
	}
	if h.Shuffle {
		normal_rand.Shuffle(len(alphabet[0]), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		normal_rand.Shuffle(len(alphabet[1]), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
	}
	//return h.reduceCUDA(alphabet)
	var orig_alpha = alphabet
	var center uint32 = 0
	for u := uint32(h.DeadlineRetry); u > 0; u-- {
		alphabet = orig_alpha
		var maxl = uint32(len(alphabet[0]))
		if len(alphabet[1]) > len(alphabet[0]) {
			maxl = uint32(len(alphabet[1]))
		}
		var maxmaxl = maxl
		var program_mut sync.Mutex
		var program [][2]uint32
		var minadd uint32 = 0
		var maxx = uint32((uint64(maxl) * uint64(maxl)) / uint64(h.Factor))
		if initMaxx != 0 {
			// forced by an external algorithm
			maxx = initMaxx
		}
		var initial = len(program) == 0
		for maxmax := maxx; maxx <= maxmax; {
			if !h.DisableProgressBar {
				const progressBarWidth = 40
				if maxmaxl > 0 {
					progress := progressBarWidth - int(maxl*progressBarWidth/maxmaxl)
					percent := 100 - int(maxl*100/maxmaxl)
					fmt.Printf("\r[%s%s] %d%% PROBLEM SIZE = %d ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent, maxx)
				}
			}
			var win_centers uint32 = 0xffffffff
			parallel.Loop(h.Threads).LoopUntil(func(nonce uint32, ender parallel.LoopStopper) (ret bool) {
				// unstucker, see below
				if int(nonce) >= h.DeadlineMs {
					return true
				}

				var centers = center ^ (nonce + minadd)
				if centers == 0xffffffff {
					return
				}

				{

					// linearly boosted loop
					var size = 0
					var par = hash.HashVectorizedParallelism()
					if par < avx.ScatterGatherVectorizedParallelism() {
						par = avx.ScatterGatherVectorizedParallelism()
					}
					var vals = make([]uint32, par, par)
					var salts = make([]uint32, par, par)
					for i := range salts {
						salts[i] = centers
					}
					var outs = make([]uint32, par, par)
					const subwords = 16
					const twobitmask = 3
					var buf = make([]uint32, (maxx+subwords-1)/subwords, (maxx+subwords-1)/subwords)
					isvBad := func(v uint32, which uint8) bool {
						wh := uint8(which)
						w0 := v / subwords
						w1 := (v % subwords) << 1
						if (buf[w0]>>w1)&twobitmask == 0 {
							size++
						}
						buf[w0] |= uint32(1+wh) << w1
						return buf[w0]&(buf[w0]>>1)&0x55555555 != 0
					}

					var minl = len(alphabet[0])
					if minl > len(alphabet[1]) {
						minl = len(alphabet[1])
					}

					minl /= par
					minl *= par

					for i := 0; i < minl; i += par {
						if ender.Load() {
							return
						}
						for j := uint8(0); j < 2; j++ {

							for k := 0; k < par; k++ {
								vals[k] = alphabet[j][i+k]
							}

							hash.HashVectorized(outs, vals, salts, maxx)

							//slices.Sort(outs)
							//compacted := slices.Compact(outs)

							if avx.ScatterGatherVectorized(outs, buf, &size, uint32(j)) {
								return
							}
						}
					}
					for j := uint8(0); j < 2; j++ {
						for i := minl; i < len(alphabet[j]); i++ {
							if ender.Load() {
								return
							}
							var v = hash.Hash(alphabet[j][i], centers, maxx)
							if isvBad(v, j) {
								return
							}
						}
					}

					// is smaller if not 2?
					if size != 2 && len(alphabet[0])+len(alphabet[1]) == size {
						return
					}
					if len(alphabet[0])+len(alphabet[1]) < size {
						panic("???")
					}
				}

				if ender.Load() {
					return
				}

				// found solution
				program_mut.Lock()
				if win_centers == 0xffffffff {
					win_centers = centers
					ret = true
				}
				program_mut.Unlock()
				return ret
			})
			// accept/reject the solution
			if win_centers == 0xffffffff {
				if initial {
					// unstucker
					maxmax = maxx
					maxx *= u
					maxx /= uint32(h.DeadlineRetry + 1)
					if maxx == 0 {
						break
					}
					continue
				} else {
					maxx++
					continue
				}
			}
			// pre-accept the solution, apply it to set
			var sets = [2]map[uint32]struct{}{make(map[uint32]struct{}), make(map[uint32]struct{})}
			for j := 0; j < 2; j++ {
				for _, v := range alphabet[j] {
					sets[j][hash.Hash(v, win_centers, maxx)] = struct{}{}
				}
			}
			// can happen in cubic verify performance that the problem doesnt shrink
			//if len(sets[0]) >= len(alphabet[0]) && len(sets[1]) >= len(alphabet[1]) {
			//	minadd = win_centers ^ center
			//	continue
			//}
			// accept the solution, set it to alphabet
			//println(maxx)
			initial = false
			program = append(program, [2]uint32{win_centers, maxx})
			for j := 0; j < 2; j++ {
				alphabet[j] = nil
				for k := range sets[j] {
					alphabet[j] = append(alphabet[j], k)
				}
			}
			maxl = uint32(len(alphabet[0]))
			if len(alphabet[1]) > len(alphabet[0]) {
				maxl = uint32(len(alphabet[1]))
			}
			if maxl == 1 && len(alphabet[0]) == len(alphabet[1]) && alphabet[0][0] == 0 && alphabet[1][0] == 1 {
				break
			}
			if maxl < untilMaxl {
				break
			}
			var sub = h.Subtractor
			if sub > maxl {
				sub = maxl - 1
			}
			newmaxx := uint32(uint64(maxx) * (uint64(maxl-sub) * uint64(maxl-sub)) / (uint64(maxl) * uint64(maxl)))
			if newmaxx >= maxx {
				minadd = win_centers ^ center
				center = win_centers // store last solution salt as the future center for xor search heuristics
			} else {
				maxmax = maxx
				maxx = newmaxx
				minadd = 0
				center = win_centers // store last solution salt as the future center for xor search heuristics
			}
			if maxx <= maxl {
				maxx = maxl
			}
			if maxx == 0 {
				break
			}
		}
		if maxl < untilMaxl {
			return program
		}
		if maxl == 1 && alphabet[0][0] == 0 && alphabet[1][0] == 1 {
			if !h.DisableProgressBar {
				const progressBarWidth = 40
				defer fmt.Printf("\r[%s] 100%% SOLUTION SIZE = %d \n", progressBar(progressBarWidth, progressBarWidth), len(program))
			}

			return program
		}
		program = nil
	}
	return nil
}

