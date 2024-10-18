package main

import "sync"
import "fmt"
import "runtime"
import "math"

import "github.com/neurlang/classifier/datasets/isvirus"
import "github.com/neurlang/classifier/layer/majpool2d"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning/avx"
import "github.com/neurlang/classifier/net/feedforward"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {
	const fanout1 = 3
	const fanout2 = 10
	const fanout3 = 3
	const fanout4 = 10
	//const fanout5 = 3
	//const fanout6 = 12
	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<fanout6)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
	net.NewLayer(fanout1*fanout2, 0) //, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
	net.NewLayer(1, 0)

	var bestErr = ^uint64(0)

	evaluate := func(final bool) {
		var mut sync.Mutex
		var percent int
		var errsum uint64

		var group = int(math.Sqrt(float64(len(isvirus.Inputs))))

		wg := [2]sync.WaitGroup{}
		var g byte
		for j := 0; j < len(isvirus.Inputs); j += group {
			for jj := 0; jj < group && jj+j < len(isvirus.Inputs); jj++ {
				wg[g].Add(1)
				go func(jjj int, gg byte) {

					var input = isvirus.Input(isvirus.Inputs[jjj])
					var output = isvirus.Output(isvirus.Outputs[jjj])

					var predicted = net.Infer3(&input).Feature(0)

					mut.Lock()

					if predicted == output.Feature(0) {
						percent++
					}
					errsum += uint64(error_abs(predicted, output.Feature(0)))

					mut.Unlock()

					wg[gg].Done()

				}(jj+j, g)

			}
			g ^= 1
			if j > 0 {
				wg[g].Wait()
			}
		}
		wg[g^1].Wait()
		success := percent * 100 / len(isvirus.Inputs)

		if bestErr == ^uint64(0) {
			println("[initial success rate]", success, "%", "with", errsum, "errors")
			// initial
			bestErr = errsum
		} else if errsum < bestErr {
			println("[improved success rate]", success, "%", "with", errsum, "errors")
			// improvement
			bestErr = errsum

			err := net.WriteCompressedWeightsToFile("output." + fmt.Sprint(success) + ".json.t.lzw")
			if err != nil {
				println(err.Error())
			}

		} else {

			println("[same success rate]", success, "%", "with", errsum, "errors")
		}
	}

	err := net.ReadCompressedWeightsFromFile("output.87.json.t.lzw")
	if err != nil {
		println(err.Error())
	}

	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		var group = int(math.Sqrt(float64(len(isvirus.Inputs))))

		wg := [2]sync.WaitGroup{}
		var g byte
		for j := 0; j < len(isvirus.Inputs); j += group {
			for jj := 0; jj < group && jj+j < len(isvirus.Inputs); jj++ {
				wg[g].Add(1)
				go func(jjj int, gg byte) {
					var input = isvirus.Input(isvirus.Inputs[jjj])
					var output = isvirus.Output(isvirus.Outputs[jjj])

					net.Tally2(&input, &output, worst, tally, func(i feedforward.FeedforwardNetworkInput) uint32 {
						return error_abs(i.Feature(0), output.Feature(0)) //< error_abs(j.Feature(0), output.Feature(0))
					})
					wg[gg].Done()

				}(jj+j, g)

			}
			g ^= 1
			if j > 0 {
				wg[g].Wait()
			}
		}
		wg[g^1].Wait()

		var h avx.HyperParameters
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

		h.AvxLanes = 16
		h.AvxSkip = 4

		h.Name = fmt.Sprint(worst)
		//h.SetLogger("solutions11.txt")

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

	for {
		shuf := net.Branch(false)
		evaluate(false)
		for worst := 0; worst < len(shuf); worst++ {
			println("training #", worst, "hastron of", len(shuf), "hashtrons total")
			trainWorst(shuf[worst])
		}
	}

}
