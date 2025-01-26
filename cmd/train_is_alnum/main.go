package main

import "sync/atomic"
import "fmt"
import "runtime"

//import "math"
//import "math/rand"
import "flag"
import "os"
import "github.com/neurlang/classifier/datasets/isalnum"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/learning"
import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/parallel"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {

	dstmodel := flag.String("dstmodel", "", "model destination .json.zlib file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0
	var difficulty uint32 = 1

	dataset := isalnum.Dataslice{}

	// a net with one neuron
	var net feedforward.FeedforwardNetwork
	net.NewLayer(1, 0)
	

	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		parallel.ForEach(dataset.Len(), 1000, func(jjj int) {
			{
				var io = dataset.Get(jjj)

				net.Tally4(&io, worst, tally, nil)
			}
		})

		var h learning.HyperParameters
		h.Threads = runtime.NumCPU()
		h.Factor = difficulty // affects the solution size

		// shuffle before solving attempts
		h.Shuffle = true
		h.Seed = true

		// restart when stuck
		h.DeadlineMs = 1000
		h.DeadlineRetry = 3

		// affects how fast is the modulo reduced
		h.Subtractor = difficulty

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
		var percent, errsum atomic.Uint64
		parallel.ForEach(dataset.Len(), 1000, func(j int) {
			{
				var io = dataset.Get(j)

				var predicted = net.Infer2(&io)

				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			}
		})
		success := 100 * int(percent.Load()) / dataset.Len()
		println("[success rate]", success, "%", "with", errsum.Load(), "errors")

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
			if difficulty > 1000 {
				println("Max accuracy or wrong data. Exiting")
				os.Exit(0)
			} else {
				difficulty++
			}
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
			if worst == len(shuf)-2 {
				evaluate()
			}
		}
	}

}
