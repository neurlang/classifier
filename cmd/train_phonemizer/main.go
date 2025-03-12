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
import "github.com/neurlang/classifier/layer/sum"
import "github.com/neurlang/classifier/layer/sochastic"
//import "github.com/neurlang/classifier/layer/parity"
import "github.com/neurlang/classifier/layer/crossattention"
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
	minpremodulo := flag.Int("minpremodulo", 0, "minpremodulo")
	maxpremodulo := flag.Int("maxpremodulo", 0, "maxpremodulo")
	maxdepth := flag.Int("maxdepth", 0, "max training depth")
	part := flag.Int("part", 0, "train on n/part-th")
	dstmodel := flag.String("dstmodel", "", "model destination .json.lzw file")
	flag.Bool("pgo", false, "enable pgo")
	boosting := flag.Bool("padspace", false, "enable padspace")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	if cleantsv == nil || *cleantsv == "" {
		println("clean tsv is mandatory")
		return
	}
	if maxdepth == nil || *maxdepth == 0 {
		println("max depth is mandatory")
		return
	}

	data := phonemizer.Split(phonemizer.NewDataset(*cleantsv, boosting != nil && *boosting))

	if len(data) == 0 {
		println("it looks like no data for this language, or language is unambiguous (no model needed)")
		return
	}

	const fanout1 = 32
	const fanout2 = 4
	const fanout3 = 3
	
	var net feedforward.FeedforwardNetwork
	net.NewLayer(fanout1*fanout2, 0)
	for i := 0; i < fanout3; i++ {
		net.NewCombiner(crossattention.MustNew(fanout1, fanout2))
		net.NewLayerPI(fanout1*fanout2, 0, 0)
		net.NewCombiner(sochastic.MustNew(fanout1*fanout2, 8*byte(i), uint32(i)))
		net.NewLayerPI(fanout1*fanout2, 0, 0)
	}
	net.NewCombiner(sochastic.MustNew(fanout1*fanout2, 32, fanout3))
	net.NewLayer(fanout1*fanout2, 0)
	net.NewCombiner(sum.MustNew([]uint{fanout1*fanout2}, 0))
	net.NewLayer(1, 0)


	trainWorst := func(worst int) func() {
		var tally = new(datasets.Tally)
		tally.Init()
		tally.SetFinalization(true)
		if premodulo != nil && *premodulo > 0 {
			tally.SetGlobalPremodulo(uint32(*premodulo))
		}
		if minpremodulo != nil && *minpremodulo > 0 && maxpremodulo != nil && *maxpremodulo > 0 {
			const span = 50 * 50
			value := (100 - improved_success_rate) * (100 - improved_success_rate)
			premodulo := value * ( *minpremodulo - *maxpremodulo ) / span + *maxpremodulo
			//println(improved_success_rate, premodulo)
			if premodulo < 2 {
				premodulo = 2
			}
			tally.SetGlobalPremodulo(uint32(premodulo))
		}
		var parts = 1
		if part != nil && *part > 1 {
			rand.Seed(time.Now().UnixNano())
			rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
			parts = *part
		}
		
		parallel.ForEach(len(data)/parts, 1000, func(jjj int) {
			{
					var io = data[jjj].V2(fanout1)

					net.Tally4(io, worst, tally, nil)
			}
		})
		
		if !tally.GetImprovementPossible() {
			return nil
		}

		fmt.Println("hashtron position:", worst, "(job size:", tally.Len(), ")")
		ptr := net.GetHashtron(worst)
		dset := tally.Dataset()
		q := quaternary.Make(dset)
		var pmod = [][2]uint32{}
		if (premodulo != nil && *premodulo > 0) || (minpremodulo != nil && *minpremodulo > 0 && maxpremodulo != nil && *maxpremodulo > 0) {
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
				var io = data[j].V2(fanout1)

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
				model := strings.ReplaceAll(*dstmodel, "weights1", "weights4")
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
		model := strings.ReplaceAll(*dstmodel, "weights1", "weights4")
	
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
			for worst := 0; worst < len(shuf) && worst < *maxdepth; worst++ {
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
