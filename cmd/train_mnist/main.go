package main

import "fmt"
import "runtime"
import "github.com/neurlang/classifier/datasets/mnist"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"

func error_abs(a,b uint16) uint16 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {
	if err := mnist.Error(); err != nil {
		panic(err.Error())
	}
	const repeat = 1
	var net FeedforwardNetwork
	const l1Dim = mnist.SmallImgSize-1
	net.NewLayer(l1Dim*l1Dim*repeat, 0)
	net.NewSumPool(3, 4, repeat)
	net.NewLayer(1, 4)
	
	//Load(net)
	
	//println(net.GetHashtron(1305))
	
	trainWorst := func(worst int) {
		var mapping = make(map[uint16]map[uint64]uint64)
		var wanted = make(map[uint32]int64)
	outer:
		for j := range mnist.InferLabels {
			var input = mnist.SmallInput(mnist.SmallInferSet[j])
			var output = uint16(mnist.InferLabels[j])
			
			switch net.GetLayer(worst) {
			case 2: // layer 2
				inter, _ := net.Forward(&input, 0, -1, 0)
				ifeature := uint16(inter.Feature(0))
				if mapping[ifeature] == nil {
					mapping[ifeature] = make(map[uint64]uint64)
				}
				mapping[ifeature][uint64(output)]++
			/*
			case 2: // layer 2
				var predicted [2]uint16
				var compute [2]byte
				
				var inter, _ = net.Forward(&input, 0, -1, 0)
				ifw := input.Feature(net.GetPosition(worst))
				if _, ok := tabu[ifw]; ok {
					continue
				}
				for neg := 0; neg < 2; neg++ {
					var inter, computed = net.Forward(inter, 2, net.GetPosition(worst), neg)
					if computed {
						compute[neg] = 1
					}
					if neg == 0 {
						if inter.Dropout(net.GetLayerPosition(2, worst)) {
							continue outer
						}
					}
					inter, _ = net.Forward(inter, 4, -1, 0)
					predicted[neg] = uint16(inter.Feature(0))
				}
				//fmt.Println(predicted, output)
				if (predicted[0] == output) && (predicted[1] == output) {
					// we are correct anyway
					continue outer
				}
				for neg := 0; neg < 2; neg++ {
					if predicted[neg] == output {
						// shift to correct output
						if _, ok := set[1^compute[neg]][ifw]; ok {
							delete(set[1^compute[neg]], ifw)
							tabu[ifw] = struct{}{}
						} else {
							set[compute[neg]][ifw] = struct{}{}
						}
						continue outer
					}
				}
				var minneg int
				for neg := 0; neg < 2; neg++ {
					if error_abs(predicted[neg], output) < error_abs(predicted[minneg], output) {
						minneg = neg
					}
				}
				
				// shift away to minneg output
				if _, ok := set[1^compute[minneg]][ifw]; ok {
					delete(set[1^compute[minneg]], ifw)
					tabu[ifw] = struct{}{}
				} else {
					set[compute[minneg]][ifw] = struct{}{}
				}
			*/
			case 0: // layer 1
				var predicted [2]uint16
				var compute [2]int8
				
				//var inter, computed = net.Forward(&input, 0, -1, 0)
				ifw := input.Feature(net.GetPosition(worst))
				//if _, ok := tabu[ifw]; ok {
				//	continue
				//}
				for neg := 0; neg < 2; neg++ {
					var inter, computed = net.Forward(&input, 0, net.GetPosition(worst), neg)
					if computed {
						compute[neg] = 1
					} else {
						compute[neg] = -1
					}
					if neg == 0 {
						if inter.Dropout(net.GetLayerPosition(0, worst)) {
							continue outer
						}
					}
					inter, _ = net.Forward(inter, 2, -1, 0)
					predicted[neg] = uint16(inter.Feature(0))
				}
				//fmt.Println(predicted, output)
				if (predicted[0] == output) && (predicted[1] == output) {
					// we are correct anyway
					continue outer
				}
				for neg := 0; neg < 2; neg++ {
					if predicted[neg] == output {
					
						wanted[ifw] += int64(compute[neg])
						// shift to correct output

						continue outer
					}
				}
				var minneg int
				for neg := 0; neg < 2; neg++ {
					if error_abs(predicted[neg], output) < error_abs(predicted[minneg], output) {
						minneg = neg
					}
				}
				
				// shift away to minneg output
				wanted[ifw] += int64(compute[minneg])
				
			}
		}

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

		h.Name = fmt.Sprint(worst)
		h.SetLogger("solutions5.txt")

		if net.GetLayer(worst) == 2 {
			var mapp datasets.Datamap
			mapp.Init()
			var suma = 0
			for k, freq := range mapping {
				var maxk uint64
				for k2 := range freq {
					if freq[k2] > freq[maxk] {
						maxk = k2
					} 
				}
				suma += int(freq[maxk])
				mapp[k] = maxk
			}
			fmt.Println(worst, len(mapp))
			htron, err := h.Training(mapp)
			if err != nil {
				panic(err.Error())
			}
			ptr := net.GetHashtron(worst)
			*ptr = *htron
		} else {
			var sett datasets.Dataset
			sett.Init()
			for value, rating := range wanted {
				if rating != 0 {
					sett[value] = rating > 0
				}
			}
			fmt.Println(worst, len(sett))
			htron, err := h.Training(sett)
			if err != nil {
				panic(err.Error())
			}
			ptr := net.GetHashtron(worst)
			*ptr = *htron
		}
	}

	for {
		var quality [2]int64
		var errsum [2]uint64
		for i, v := range [2][]byte{mnist.TrainLabels, mnist.InferLabels} {
			for j := range v {
				var input = mnist.SmallInput(mnist.SmallTrainSet[j])
				var output = uint16(mnist.TrainLabels[j])
				if i == 1 {
					input = mnist.SmallInput(mnist.SmallInferSet[j])
					output = uint16(mnist.InferLabels[j])
				}
				var predicted uint16
				{
					//inter, _ := net.Forward(&input, 0, -1, 0)
					inter, _ := net.Forward(&input, 0, -1, 0)
					inter, _ = net.Forward(inter, 2, -1, 0)
					predicted = uint16(inter.Feature(0))

				}
				if predicted == output {
					quality[i]++
				} else {
					quality[i]--
				}
				errsum[i] += uint64(error_abs(predicted, output))
			}
		}
		println(quality[0], errsum[0], quality[1], errsum[1])
		mnist.ShuffleInfer()
		for worst := 0; worst < net.Len(); worst++ {
			trainWorst(worst)
		}

	}
}
