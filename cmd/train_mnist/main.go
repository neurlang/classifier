package main

import "sync"
import "fmt"
import "runtime"
//import "math"
//import "math/rand"
import "flag"
import "os"
import "github.com/neurlang/classifier/datasets/mnist"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"
//import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"
//import "github.com/neurlang/classifier/layer/full"
//import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/net/feedforward"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {

	dstmodel := flag.String("dstmodel", "", "model destination .json.lzw file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	if err := mnist.Error(); err != nil {
		panic(err.Error())
	}

	const fanout1 = 3
	const fanout2 = 5
	const fanout3 = 3
	const fanout4 = 5
	//const fanout5 = 3
	//const fanout6 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<fanout6)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
	net.NewLayer(1, 0)



	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		const group = 500
		for j := 0; j < len(mnist.TrainLabels); j += group {
			wg := sync.WaitGroup{}
			for jj := 0; jj < group && jj+j < len(mnist.TrainLabels); jj++ {
				wg.Add(1)
				go func(jjj int) {
					var input = mnist.SmallInput(mnist.SmallTrainSet[jjj])
					var output = feedforward.SingleValue(mnist.TrainLabels[jjj]&1)

					net.Tally2(&input, &output, worst, tally, func(i feedforward.FeedforwardNetworkInput) uint32 {
						return error_abs(i.Feature(0), output.Feature(0)) //< error_abs(j.Feature(0), output.Feature(0))
					})
					wg.Done()

				}(jj + j)

			}
			wg.Wait()
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

		h.Name = fmt.Sprint(worst)
		//h.SetLogger("solutions11.txt")
		
		//h.AvxLanes = 16
		//h.AvxSkip = 4

		fmt.Println("hashtron position:", worst, "(job size:", tally.Len(), ")")

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
		var percent int
		var errsum uint64
		for j := range mnist.TrainLabels {
			{
				var input = mnist.SmallInput(mnist.SmallTrainSet[j])
				var output = feedforward.SingleValue(mnist.TrainLabels[j]&1)

				var predicted = net.Infer(&input).Feature(0)
				if predicted == output.Feature(0) {
					percent++
				}
				errsum += uint64(error_abs(predicted, output.Feature(0)))
			}
		}
		success := percent * 100 / len(mnist.TrainLabels)
		println("[success rate]", success, "%", "with", errsum, "errors")

		if dstmodel == nil || *dstmodel == "" {
			err := net.WriteZlibWeightsToFile("output." + fmt.Sprint(success) + ".json.t.zlib")
			if err != nil {
				println(err.Error())
			}
		}

		if dstmodel != nil && len(*dstmodel) > 0 && improved_success_rate < success {
			if improved_success_rate > 0 {
				err := net.WriteZlibWeightsToFile(*dstmodel)
				if err != nil {
					println(err.Error())
				}
			}
			improved_success_rate = success
		}

		if success == 100 {
			println("Max accuracy or wrong data. Exiting")
			os.Exit(0)
		}
	}
	if resume != nil && *resume && dstmodel != nil {
		net.ReadZlibWeightsFromFile(*dstmodel)
	}
	for {
		shuf := net.Branch(false)
		evaluate()
		for worst := 0; worst < len(shuf); worst++ {
			println("training #", worst, "hastron of", len(shuf), "hashtrons total")
			trainWorst(shuf[worst])
			if worst == len(shuf) - 2 {
				evaluate()
			}
		}
	}



}
