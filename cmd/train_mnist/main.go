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

	//net.NewLayerP(121*121, 0, 1031)
	//net.NewCombiner(conv2d.MustNew(121, 121, 6, 6, 1))
	//net.NewLayer(116*116, 0)
	//net.NewCombiner(majpool2d.MustNew(116, 116, 2, 2, 1, 1, 1))
	//net.NewLayerP(58*58, 0, 1031)
	//net.NewCombiner(conv2d.MustNew(58, 58, 5, 5, 1))
	//net.NewLayer(54*54, 0)
	//net.NewCombiner(majpool2d.MustNew(54, 54, 2, 2, 1, 1, 1))

	net.NewLayer(27*27, 0) //2053
	net.NewCombiner(conv2d.MustNew2(27, 27, 16, 16, 1, 4))
	net.NewLayer(12*12, 0)
	net.NewCombiner(majpool2d.MustNew(4, 4, 3, 3, 4, 4, 1))
	net.NewLayerP(1, 4, 2053) //2053

	//Load(net)

	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(true)

		const group = 1000
		for j := 0; j < len(mnist.InferLabels); j+=group {
			wg := sync.WaitGroup{}
			for jj := 0; jj < group && jj + j < len(mnist.InferLabels); jj++ {
				wg.Add(1)
				go func(jjj int) {
					var input = mnist.Input(mnist.InferSet[jjj])
					var output = feedforward.SingleValue(mnist.InferLabels[jjj])

					net.Tally2(&input, &output, worst, tally, func(i feedforward.FeedforwardNetworkInput) uint32 {
						return error_abs(i.Feature(0), output.Feature(0)) //< error_abs(j.Feature(0), output.Feature(0))
					})
					wg.Done()

				}(jj+j)

			}
			wg.Wait()
		}

		if !tally.GetImprovementPossible() {
			return
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
		h.InitialLimit = 1000 + 4*tally.Len()
		h.EndWhenSolved = true

		//h.CuErase = true
		//h.CuCutoff = 400
		//h.CuMemoryPortion = 100
		//h.CuMemoryBytes = 1000000000

		h.Name = fmt.Sprint(worst)
		h.SetLogger("solutions19.txt")

		fmt.Println(worst, tally.Len())

		htron, err := h.Training(tally)
		if err != nil {
			panic(err.Error())
		}
		ptr := net.GetHashtron(worst)
		*ptr = *htron

		tally.Free()
		runtime.GC()
	}
	evaluate := func() {
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
	}
	//trainWorst(985)
	for {
		evaluate()
		shuf := net.Shuffle(true)
		for worst := 0; worst < net.Len(); worst++ {
			println("training", worst)
			trainWorst(shuf[worst])
			if worst == 0 {
				evaluate()
			}
		}
		net.SetLayersP(0)
	}
}
