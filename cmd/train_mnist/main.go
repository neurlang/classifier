package main

import "sync"
import "fmt"
import "runtime"
import "github.com/neurlang/classifier/datasets/mnist"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"
import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"
import "github.com/neurlang/classifier/net/feedforward"

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
	var net feedforward.FeedforwardNetwork
	const l0Dim = mnist.ImgSize-1
	const l1Dim = mnist.SmallImgSize-1
	net.NewLayer(l0Dim*l0Dim, 0)
	net.NewCombiner(conv2d.MustNew(27, 27, 16, 16, 1))
	net.NewLayer(l1Dim*l1Dim, 0)
	net.NewCombiner(majpool2d.MustNew(3, 3, 4, 4, 1))
	net.NewLayer(1, 4)
	
	//Load(net)
	
	//println(net.GetHashtron(1305))
	
	trainWorst := func(worst int) {
		var tally datasets.Tally
		tally.Init()
		wg := sync.WaitGroup{}

		for jj := range mnist.InferLabels {

			wg.Add(1)
			go func(jj int) {
			var input = mnist.Input(mnist.InferSet[jj])
			var output = uint16(mnist.InferLabels[jj])



			switch net.GetLayer(worst) {
			case 4: // layer 2
				inter, _ := net.Forward(&input, 0, -1, 0)
				inter, _ = net.Forward(inter, 2, -1, 0)

				ifeature := uint16(inter.Feature(0))

				tally.AddToMapping(ifeature, uint64(output))

			case 2: // layer 1
				var predicted [2]uint16
				var compute [2]int8
				
				var inter, _ = net.Forward(&input, 0, -1, 0)
				ifw := input.Feature(net.GetPosition(worst))
				for neg := 0; neg < 2; neg++ {
					inter, computed := net.Forward(inter, 2, net.GetPosition(worst), neg)
					if computed {
						compute[neg] = 1
					} else {
						compute[neg] = -1
					}
					if neg == 0 {
						if inter.Disregard(net.GetLayerPosition(2, worst)) {
							goto end
						}
					}
					inter, _ = net.Forward(inter, 4, -1, 0)
					predicted[neg] = uint16(inter.Feature(0))
				}
				//fmt.Println(predicted, output)
				if (predicted[0] == output) && (predicted[1] == output) {
					// we are correct anyway
					goto end
				}
				for neg := 0; neg < 2; neg++ {
					if predicted[neg] == output {

						tally.AddToCorrect(ifw, compute[neg])
						// shift to correct output

						goto end
					}
				}
				var minneg int
				for neg := 0; neg < 2; neg++ {
					if error_abs(predicted[neg], output) < error_abs(predicted[minneg], output) {
						minneg = neg
					}
				}
				
				// shift away to minneg output
				tally.AddToImprove(ifw, compute[minneg])
				
			case 0: // layer 0
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
						if inter.Disregard(net.GetLayerPosition(0, worst)) {
							goto end
						}
					}
					inter, _ = net.Forward(inter, 2, -1, 0)
					inter, _ = net.Forward(inter, 4, -1, 0)
					predicted[neg] = uint16(inter.Feature(0))
				}
				//fmt.Println(predicted, output)
				if (predicted[0] == output) && (predicted[1] == output) {
					// we are correct anyway
					goto end
				}
				for neg := 0; neg < 2; neg++ {
					if predicted[neg] == output {
					
						tally.AddToCorrect(ifw, compute[neg])
						// shift to correct output

						goto end
					}
				}
				var minneg int
				for neg := 0; neg < 2; neg++ {
					if error_abs(predicted[neg], output) < error_abs(predicted[minneg], output) {
						minneg = neg
					}
				}
				
				// shift away to minneg output
				tally.AddToImprove(ifw, compute[minneg])
				
			}
end:
				wg.Done()

			}(jj)
		}

		wg.Wait()

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
		h.InitialLimit = 1000+4*tally.Len()
		h.EndWhenSolved = true

		h.Name = fmt.Sprint(worst)
		h.SetLogger("solutions10.txt")

		fmt.Println(worst, tally.Len())

		htron, err := h.Training(&tally)
		if err != nil {
			panic(err.Error())
		}
		ptr := net.GetHashtron(worst)
		*ptr = *htron

		tally.Free()
		runtime.GC()
	}

	for {
		var quality [2]int64
		var errsum [2]uint64
		for i, v := range [2][]byte{mnist.TrainLabels, mnist.InferLabels} {
			for j := range v {
				var input = mnist.Input(mnist.TrainSet[j])
				var output = uint16(mnist.TrainLabels[j])
				if i == 1 {
					input = mnist.Input(mnist.InferSet[j])
					output = uint16(mnist.InferLabels[j])
				}
				var predicted uint16
				{
					//inter, _ := net.Forward(&input, 0, -1, 0)
					inter, _ := net.Forward(&input, 0, -1, 0)
					inter, _ = net.Forward(inter, 2, -1, 0)
					inter, _ = net.Forward(inter, 4, -1, 0)
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
		//mnist.ShuffleInfer()
		for worst := 0; worst < net.Len(); worst++ {
			trainWorst(worst)
		}

	}
}
