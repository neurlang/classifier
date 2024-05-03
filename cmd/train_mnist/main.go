package main

import "fmt"
import "runtime"
import "github.com/neurlang/classifier/datasets/mnist"
import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"

type Input [mnist.SmallImgSize*mnist.SmallImgSize]byte
func (i *Input) Feature(n int) uint32 {
	return uint32(i[n]) | uint32(i[n+1]) | uint32(i[n+mnist.SmallImgSize]) | uint32(i[n+1+mnist.SmallImgSize])
}

type SumPool [4*4][3*3]bool
func (s *SumPool) Put(n int, v uint16) {
	x := n / 12
	y := n % 12
	xx := x / 3
	xy := x % 3
	yx := y / 3
	yy := y % 3
	s[xx+4*yx][xy+3*yy] = (v & 1) != 0
}
func (s *SumPool) Dropout(n int) bool {
	x := n / 12
	y := n % 12
	xx := x / 3
	yx := y / 3
	var w = byte(9)
	for _, v := range s[xx+4*yx] {
		if v {
			w++
		} else {
			w--
		}
	}
	if w == 8 || w == 10 {
		return false
	} else {
		return true
	}
}
func (s *SumPool) Feature(_ int) (o uint32) {
	for n := 0; n < 16; n++ {
		var w = byte(9)
		for _, v := range s[n] {
			if v {
				w++
			} else {
				w--
			}
		}
		if w > 9 {
			o |= 1 << n
		}
	}
	return 
}

func main() {
	if err := mnist.Error(); err != nil {
		panic(err.Error())
	}
	const l1Dim = mnist.SmallImgSize-1
	var L1 [l1Dim*l1Dim]hashtron.Hashtron
	for i := range L1 {
		h, _ := hashtron.New(nil, 0)
		L1[i] = *h
	}
	var L2 [1]hashtron.Hashtron
	for i := range L2 {
		h, _ := hashtron.New(nil, 4)
		L2[i] = *h
	}
	
	trainWorst := func(worst int) {
		var set datasets.SplittedDataset
		set.Init()
		var mapp datasets.Datamap
		mapp.Init()
		var tabu = make(map[uint32]struct{})
	outer:
		for j := range mnist.TrainLabels {
			var input = Input(mnist.SmallTrainSet[j])
			var output = uint16(mnist.TrainLabels[j])
			if worst == l1Dim*l1Dim { // last layer
				var hidden SumPool
				for i := range L1 {
					hidden.Put(i, L1[i].Forward(input.Feature(i), false))
				}
				mapp[uint16(hidden.Feature(0))] = output
			} else {
				var predicted [2]uint16
				var computed [2]uint16
				ifw := input.Feature(worst)
				if _, ok := tabu[ifw]; ok {
					continue
				}
				for neg := 0; neg < 2; neg++ {
					var hidden SumPool
					for i := range L1 {
						var bit = L1[i].Forward(input.Feature(i), (i == worst) && (neg == 1))
						if i == worst {
							computed[neg] = bit & 1
						}
						hidden.Put(i, bit)
					}
					if neg == 0 {
						if hidden.Dropout(worst) {
							continue outer
						}
					}
					for i := range L2 {
						predicted[neg] = L2[i].Forward(hidden.Feature(i), false)
					}
				}

				//fmt.Println(predicted, output)
				if (predicted[0] != output) == (predicted[1] != output) {
					if len(set[1]) < len(set[0]) {
						if _, ok := set[0][ifw]; !ok {
							set[1][ifw] = struct{}{}
						}
					} else {
						if _, ok := set[1][ifw]; !ok {
							set[0][ifw] = struct{}{}
						}
					}
					// doesn't matter
				} else if predicted[0] != output {
					if _, ok := set[1^computed[0]][ifw]; ok {
						delete(set[1^computed[0]], ifw)
						tabu[ifw] = struct{}{}
					} else {
						set[computed[0]][ifw] = struct{}{}
					}
				} else if predicted[1] != output {
					if _, ok := set[1^computed[1]][ifw]; ok {
						delete(set[1^computed[1]], ifw)
						tabu[ifw] = struct{}{}
					} else {
						set[computed[1]][ifw] = struct{}{}
					}
				}
			}
		}
		fmt.Println(worst, len(set[0]), len(set[1]))
		var h learning.HyperParameters
		h.Threads = runtime.NumCPU()
		h.Factor = 1 // affects the solution size

		// shuffle before solving attempts
		h.Shuffle = true
		h.Seed = true

		// restart when stuck
		h.DeadlineMs = 1000
		h.DeadlineRetry = 3

		// affects how fast is the modulo reduced
		h.Subtractor = 1

		// reduce Backtracking printing on the log
		h.Printer = 70

		// save any solution to disk
		h.InitialLimit = 9999999999
		h.EndWhenSolved = true


		if worst == l1Dim*l1Dim {
			htron, err := h.Training(mapp)
			if err != nil {
				panic(err.Error())
			}
			L2[0] = *htron
		} else {
			htron, err := h.Training(set)
			if err != nil {
				panic(err.Error())
			}
			L1[worst] = *htron
		}
	}

	for {
		mnist.ShuffleTrain()
		for worst := 0; worst <= l1Dim*l1Dim; worst++ {
			trainWorst(worst)
		}
		var quality [2]int64
		for i, v := range [2][]byte{mnist.TrainLabels, mnist.InferLabels} {
			for j := range v {
				//for i := 0; i < mnist.ImgSize; i++ {
				//	fmt.Printf("%x\n", mnist.TrainSet[j][i*mnist.ImgSize:(i+1)*mnist.ImgSize])
				//}
				//for i := 0; i < mnist.SmallImgSize; i++ {
				//	fmt.Printf("%x\n", mnist.SmallTrainSet[j][i*mnist.SmallImgSize:(i+1)*mnist.SmallImgSize])
				//}
				var input = Input(mnist.SmallTrainSet[j])
				var output = uint16(mnist.TrainLabels[j])
				if i == 1 {
					input = Input(mnist.SmallInferSet[j])
					output = uint16(mnist.InferLabels[j])
				}
				var predicted [2]uint16
				{
					var hidden SumPool
					for i := range L1 {
						hidden.Put(i, L1[i].Forward(input.Feature(i), false))
					}
					for i := range L2 {
						predicted[0] = L2[i].Forward(hidden.Feature(i), false)
					}
				}
				if predicted[0] == output {
					quality[i]++
				} else {
					quality[i]--
				}
			}
		}
		println(quality[0], quality[1])
	}
}
