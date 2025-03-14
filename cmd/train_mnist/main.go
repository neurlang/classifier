package main

import "sync/atomic"
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
//import "github.com/neurlang/classifier/layer/sochastic"
//import "github.com/neurlang/classifier/layer/sum"
import "github.com/neurlang/classifier/layer/full"
//import "github.com/neurlang/classifier/hashtron"
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
	srcmodel := flag.String("srcmodel", "", "model source .json.zlib file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0
	
	const classes = 10

	dataslice, _, _, _, err := mnist.New()
	if err != nil {
		panic(err.Error())
	}
	

	const fanout1 = 1
	const fanout2 = 5
	const fanout3 = 1
	const fanout4 = 4
	const fanout5 = 1
	const fanout6 = 4
	//const fanout7 = 1
	//const fanout8 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6*fanout7*fanout8, 0, 1<<fanout8)
	//net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6*fanout8, 1, fanout7, 1, fanout8, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<(fanout6*fanout6*2/3))
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<(fanout4*fanout4*2/3))
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2, 0, 1<<(fanout2*fanout2*2/3))
	net.NewCombiner(full.MustNew(fanout2, 1, 1))
	//net.NewCombiner(majpool2d.MustNew2(fanout2, 1, fanout1, 1, fanout2, 1, 1, 0))
	//net.NewLayerP(1, 4, 1<<16)


	trainWorst := func(worst int) {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(false)

		parallel.ForEach(len(dataslice), 1000, func(jjj int) {
			{
				var io = dataslice[jjj]

				net.Tally4(&io, worst, tally, func(actual uint32, expected uint32, mask uint32) uint32 {				
					//return error_abs(actual % classes, expected % classes)
					if actual % classes == expected % classes {
						return 0
					}
					return 1
				})
			}
		})

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
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(dataslice), 1000, func(j int) {
			{
				var io = dataslice[j]

				var predicted = net.Infer2(&io) % classes
				//println(predicted, io.Output())
				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			}
		})
		success := 100 * int(percent.Load()) / len(dataslice)
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
			println("Max accuracy or wrong data. Exiting")
			os.Exit(0)
		}
	}
	if resume != nil && *resume && dstmodel != nil {
		net.ReadZlibWeightsFromFile(*dstmodel)
	}
	if resume != nil && *resume && srcmodel != nil {
		net.ReadZlibWeightsFromFile(*srcmodel)
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
