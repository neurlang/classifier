package main

import "os"
import "sync/atomic"
import "fmt"
import "runtime"
import "flag"

import "github.com/neurlang/classifier/datasets/phonemizer"
import "github.com/neurlang/classifier/layer/majpool2d"
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
	cleantsv := flag.String("cleantsv", "", "clean tsv dataset for the language")
	dstmodel := flag.String("dstmodel", "", "model destination .json.lzw file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	if cleantsv == nil || *cleantsv == "" {
		println("clean tsv is mandatory")
		return
	}

	data := phonemizer.Split(phonemizer.NewDataset(*cleantsv))

	if len(data) == 0 {
		println("it looks like no data for this language, or language is unambiguous (no model needed)")
		return
	}

	const fanout1 = 1
	const fanout2 = 5
	const fanout3 = 3
	const fanout4 = 5
	//const fanout5 = 1
	//const fanout6 = 4
	//const fanout7 = 1
	//const fanout8 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6*fanout7*fanout8, 0, 1<<fanout8)
	//net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6*fanout8, 1, fanout7, 1, fanout8, 1, 1, 0))
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<(fanout6*fanout6*2/3))
	//net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<13)
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0))
	net.NewLayer(fanout1*fanout2, 0)
	//net.NewCombiner(full.MustNew(fanout2, 1, 1))
	net.NewCombiner(majpool2d.MustNew2(fanout2, 1, fanout1, 1, fanout2, 1, 1, 0))
	net.NewLayer(1, 0)



	trainWorst := func(worst int) bool {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(true)

		parallel.ForEach(len(data), 1000, func(jjj int) {
			{
					var io = data[jjj]

					net.Tally4(&io, worst, tally, nil)
			}
		})
		
		if !tally.GetImprovementPossible() {
			return false
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
		
		return true
	}
	evaluate := func() (int, [32]byte) {
		var h = parallel.NewUint16Hasher(len(data))
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(data), 1000, func(j int) {
			{
				var io = data[j]

				var predicted = net.Infer2(&io) & 1
				
				h.MustPutUint16(j, predicted)
				
				if predicted == io.Output() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			}
		})
		success := 100 * int(percent.Load()) / len(data)
		println("[success rate]", success, "%", "with", errsum.Load(), "errors")

		if dstmodel == nil || *dstmodel == "" {
			err := net.WriteZlibWeightsToFile("output." + fmt.Sprint(success) + ".json.t.lzw")
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
		return success, h.Sum()
	}
	if resume != nil && *resume && dstmodel != nil {
		err := net.ReadZlibWeightsFromFile(*dstmodel)
		if err != nil {
			println(err.Error())
		}
	}
	var m = parallel.NewMoveSet()
	var success, state = evaluate()
	fmt.Printf("%x\n", state)
	for infloop := 0; infloop < net.Len(); infloop++ {
		shuf := net.Branch(false)
		if m.Exists(state, shuf[0], byte(success)) {
			continue
		}
		for worst := 0; worst < len(shuf); worst++ {
			println("training #", worst, "hastron of", len(shuf), "hashtrons total")
			if trainWorst(shuf[worst]) {
				infloop = 0
				success, state = evaluate()
			} else if worst == 0 {
				break
			}
			fmt.Printf("%x\n", state)
			m.Insert(state, shuf[worst], byte(success))
			if worst != len(shuf)-1 {
				if m.Exists(state, shuf[worst+1], byte(success)) {
					break
				}
			}
		}
	}
	println("Infinite loop - algorithm stuck in local minima. Exiting")
}
