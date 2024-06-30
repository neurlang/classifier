package main

import "sync"
import "fmt"
import "runtime"
import "flag"

import "github.com/neurlang/classifier/datasets/phonemizer"
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

func main() {
	cleantsv := flag.String("cleantsv", "", "clean tsv dataset for the language")
	flag.Parse()

	if cleantsv == nil || *cleantsv == "" {
		println("clean tsv is mandatory")
		return
	}

	datakeys, datavalues := phonemizer.Split(phonemizer.NewDataset(*cleantsv))

	const fanout1 = 3
	const fanout2 = 12
	//const fanout3 = 3
	//const fanout4 = 10
	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1033)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
	net.NewLayer(1, 0)

	//net.ReadCompressedWeightsFromFile("output.94.json.t.lzw")

	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		const group = 500
		for j := 0; j < len(datakeys); j += group {
			wg := sync.WaitGroup{}
			for jj := 0; jj < group && jj+j < len(datakeys); jj++ {
				wg.Add(1)
				go func(jjj int) {
					var input = phonemizer.Sample(datakeys[jjj])
					var output = phonemizer.Output(datavalues[jjj])

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
		for j := range datakeys {
			{
				var input = phonemizer.Sample(datakeys[j])
				var output = phonemizer.Output(datavalues[j])

				var predicted = net.Infer(&input).Feature(0)
				if predicted == output.Feature(0) {
					percent++
				}
				errsum += uint64(error_abs(predicted, output.Feature(0)))
			}
		}
		success := percent * 100 / len(datakeys)
		println("[success rate]", success, "%", "with", errsum, "errors")

		err := net.WriteCompressedWeightsToFile("output." + fmt.Sprint(success) + ".json.t.lzw")
		if err != nil {
			println(err.Error())
		}
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
