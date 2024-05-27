package main

import "sync"
import "fmt"
import "runtime"

import "github.com/neurlang/classifier/datasets/isvirus"
import "github.com/neurlang/classifier/layer/majpool2d"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"
import "github.com/neurlang/classifier/net/feedforward"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func init() {
	//isvirus.Balance()
}

func main() {
	const fanout = 10
	var net feedforward.FeedforwardNetwork
	//net.NewLayer(fanout*fanout*fanout*fanout, 0)
	//net.NewCombiner(majpool2d.MustNew(fanout*fanout, 1, fanout*fanout, 1, 1, 1, 1))
	net.NewLayer(fanout*fanout, 0)
	net.NewCombiner(majpool2d.MustNew(fanout, 1, fanout, 1, fanout, 1, 1))
	net.NewLayer(1, 0)

	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		const group = 500
		for j := 0; j < len(isvirus.Inputs); j += group {
			wg := sync.WaitGroup{}
			for jj := 0; jj < group && jj+j < len(isvirus.Inputs); jj++ {
				wg.Add(1)
				go func(jjj int) {
					var input = isvirus.Input(isvirus.Inputs[jjj])
					var output = isvirus.Output(isvirus.Outputs[jjj])

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
		h.SetLogger("solutions11.txt")

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
		for j := range isvirus.Inputs {
			{
				var input = isvirus.Input(isvirus.Inputs[j])
				var output = isvirus.Output(isvirus.Outputs[j])

				var predicted = net.Infer(&input).Feature(0)
				if predicted == output.Feature(0) {
					percent++
				}
				errsum += uint64(error_abs(predicted, output.Feature(0)))
			}
		}
		println("[success rate]", percent*100/len(isvirus.Inputs), "%", "with", errsum, "errors")
	}

	for {
		shuf := net.Shuffle(true)
		evaluate()
		for worst := 0; worst < len(shuf); worst++ {
			println("training #", worst, "hastron of", len(shuf), "hashtrons total")
			trainWorst(shuf[worst])
			if worst == 0 {
				evaluate()
			}
		}
	}

}
