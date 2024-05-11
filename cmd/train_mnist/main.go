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

func error_abs(a, b uint32) uint32 {
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
	const l0Dim = mnist.ImgSize - 1
	const l1Dim = mnist.SmallImgSize - 1
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
				var output = feedforward.SingleValue(mnist.InferLabels[jj])

				net.Tally(&input, &output, worst, tally, func(i, j feedforward.FeedforwardNetworkInput) bool {
					return error_abs(i.Feature(0), output.Feature(0)) < error_abs(j.Feature(0), output.Feature(0))
				})
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
		h.InitialLimit = 1000 + 4*tally.Len()
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
				var output = uint32(mnist.TrainLabels[j])
				if i == 1 {
					input = mnist.Input(mnist.InferSet[j])
					output = uint32(mnist.InferLabels[j])
				}
				var predicted = net.Infer(&input).Feature(0)
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
