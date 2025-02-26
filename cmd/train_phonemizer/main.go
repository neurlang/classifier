package main

import "os"
import "sync/atomic"
import "fmt"
import "runtime"
import "flag"
import "strings"
import "math/rand"
import "time"

import "github.com/neurlang/classifier/datasets/phonemizer"
//import "github.com/neurlang/classifier/layer/majpool2d"
//import "github.com/neurlang/classifier/layer/sum"
//import "github.com/neurlang/classifier/layer/sochastic"
import "github.com/neurlang/classifier/layer/parity"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hashtron"
//import "github.com/neurlang/classifier/learning"
import "github.com/neurlang/quaternary"
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
	premodulo := flag.Int("premodulo", 0, "premodulo")
	part := flag.Int("part", 0, "train on n/part-th")
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
/*



	net.NewLayer(fanout1*fanout2*fanout3, 0)
	net.NewCombiner(sochastic.MustNew(fanout1*fanout2*fanout3, 32, 1))
	net.NewLayer(fanout1*fanout2, 0)

	*/
	const fanout1 = 5
	var net feedforward.FeedforwardNetwork
	//net.NewLayer(fanout1, 0)
	//net.NewCombiner(sochastic.MustNew(fanout1, 32, 0))
	net.NewLayer(fanout1, 0)
	net.NewCombiner(parity.MustNew(fanout1))
	net.NewLayer(1, 0)
	/*
	net.NewCombiner(sochastic.MustNew(1, 32, 0))
	net.NewLayerPI(1, 0, 0)
	net.NewCombiner(sochastic.MustNew(1, 32, 0))
	net.NewLayerPI(1, 0, 0)
	net.NewCombiner(sochastic.MustNew(1, 1, 0))
*/
/*
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
*/

	trainWorst := func(worst int) func() {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(true)
		if premodulo != nil && *premodulo > 0 {
			tally.SetGlobalPremodulo(uint32(*premodulo))
		}
		
		var parts = 1
		if part != nil && *part > 1 {
			rand.Seed(time.Now().UnixNano())
			rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
			parts = *part
		}
		
		parallel.ForEach(len(data)/parts, 1000, func(jjj int) {
			{
					var io = data[jjj].V1()

					net.Tally4(io, worst, tally, nil)
			}
		})
		
		if !tally.GetImprovementPossible() {
			return nil
		}
/*
		var h learning.HyperParameters
		h.Threads = runtime.NumCPU()
		h.Factor = 1 // affects the solution size

		// shuffle before solving attempts
		h.Shuffle = true
		h.Seed = true

		// restart when stuck
		h.DeadlineMs = 1000
		h.DeadlineRetry = 10

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
*/
		fmt.Println("hashtron position:", worst, "(job size:", tally.Len(), ")")
		ptr := net.GetHashtron(worst)
		dset := tally.Dataset()
		q := quaternary.Make(dset)
		var pmod = [][2]uint32{}
		if premodulo != nil && *premodulo > 0 {
			pmod = [][2]uint32{tally.GetGlobalSaltPremodulo()}
		}
		htron, err := hashtron.New(pmod, ptr.Bits(), []byte(q))
		if err != nil {
			panic(err.Error())
		}
		var backup = *ptr
		*ptr = *htron

		tally.Free()
		runtime.GC()

		return func(){
			*ptr = backup
		}
	}
	evaluate := func() (int, [32]byte) {
		var parts = 1
		if part != nil && *part > 1 {
			parts = *part
		}
		var h = parallel.NewUint16Hasher(len(data))
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(data)/parts, 1000, func(j int) {
			{
				var io = data[j].V1()

				var predicted = net.Infer2(io) & 1
				
				h.MustPutUint16(j, predicted)
				
				if predicted == io.Output() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			}
		})
		success := 100 * int(percent.Load()) / (len(data)/parts)
		println("[success rate]", success, "%", "with", uint64(parts) * errsum.Load(), "errors")

		if dstmodel == nil || *dstmodel == "" {
			err := net.WriteZlibWeightsToFile("output." + fmt.Sprint(success) + ".json.t.lzw")
			if err != nil {
				println(err.Error())
			}
		}

		if dstmodel != nil && len(*dstmodel) > 0 && improved_success_rate < success {
			if improved_success_rate > 0 {
				model := strings.ReplaceAll(*dstmodel, "weights1", "weights2")
				err := net.WriteZlibWeightsToFile(model)
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
		model := strings.ReplaceAll(*dstmodel, "weights1", "weights2")
	
		err := net.ReadZlibWeightsFromFile(model)
		if err != nil {
			println(err.Error())
		}
	}
	var m = parallel.NewMoveSet()
	var success, state = evaluate()
	var default_backoff = func() {
		println("Infinite loop - algorithm stuck in local minimum. Exiting")
		os.Exit(0)
	}
	backoff := default_backoff
	var local_minimums = make(map[[32]byte]struct{})
	fmt.Printf("%x\n", state)
	for {
		for infloop := 0; infloop < net.Len(); infloop++ {
			shuf := net.Branch(false)
			if m.Exists(state, shuf[0], byte(success)) {
				continue
			}
			for worst := 0; worst < len(shuf); worst++ {
				println("training #", worst, "hastron of", len(shuf), "hashtrons total")
				if this_backoff := trainWorst(shuf[worst]); this_backoff != nil {
					infloop = -1
					this_success, this_state := evaluate()
					if _, bad := local_minimums[this_state]; bad {
						this_backoff()
						break
					} else {
						backoff, success, state = this_backoff, this_success, this_state
					}
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
		local_minimums[state] = struct{}{}
		backoff()
		backoff = default_backoff
		success, state = evaluate()
	}
}
