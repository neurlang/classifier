package main

import "sync/atomic"
import "flag"

import "github.com/neurlang/classifier/datasets/phonemizer_multi"

//import "github.com/neurlang/classifier/layer/majpool2d"
import "github.com/neurlang/classifier/layer/sum"
import "github.com/neurlang/classifier/layer/sochastic"

//import "github.com/neurlang/classifier/layer/parity"
import "github.com/neurlang/classifier/layer/crossattention"
import "github.com/neurlang/classifier/datasets"

//import "github.com/neurlang/classifier/learning"
import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/parallel"
import "github.com/neurlang/classifier/trainer"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {
	langdir := flag.String("langdir", "", "lang directory for the language")
	premodulo := flag.Int("premodulo", 0, "premodulo")
	minpremodulo := flag.Int("minpremodulo", 0, "minpremodulo")
	maxpremodulo := flag.Int("maxpremodulo", 0, "maxpremodulo")
	maxdepth := flag.Int("maxdepth", 0, "max training depth")
	dstmodel := flag.String("dstmodel", "", "model destination .json.lzw file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	if langdir == nil || *langdir == "" {
		println("clean tsv is mandatory")
		return
	}
	if maxdepth == nil || *maxdepth == 0 {
		println("max depth is mandatory")
		return
	}

	data := phonemizer_multi.NewDataset(*langdir)

	if len(data) == 0 {
		println("it looks like no data for this language, or language is unambiguous (no model needed)")
		return
	}

	const fanout1 = 24
	const fanout2 = 1
	const fanout3 = 8

	var net feedforward.FeedforwardNetwork
	net.NewLayer(fanout1*fanout2, 0)
	for i := 0; i < fanout3; i++ {
		if i == 0 {
			net.NewCombiner(crossattention.MustNew3(fanout1, fanout2))
		} else {
			net.NewCombiner(crossattention.MustNew3(fanout1, fanout2))
		}
		net.NewLayerPI(fanout1*fanout2, 0, 0)
		net.NewCombiner(sochastic.MustNew(fanout1*fanout2, 4*byte(i), uint32(i)))
		net.NewLayerPI(fanout1*fanout2, 0, 0)
	}
	net.NewCombiner(sochastic.MustNew(fanout1*fanout2, 32, fanout3))
	net.NewLayer(fanout1*fanout2, 0)
	net.NewCombiner(sum.MustNew([]uint{fanout1 * fanout2}, 0))
	net.NewLayer(1, 0)

	trainWorst := trainer.NewTrainWorstFunc(net, minpremodulo, premodulo, maxpremodulo,
		func(worst []int, tally datasets.AnyTally) {
			parallel.ForEach(len(data), 1000, func(jjj int) {
				for i := 0; i < fanout1; i++ {
					sample := data[jjj].V2(fanout1, i)
					for j := 0; j < sample.Len(); j++ {
						io := sample.IO(j)
						net.AnyTally(io, worst, tally, nil)
					}
				}
			})
		})
	evaluate := trainer.NewEvaluateFunc(net, len(data), 99, &improved_success_rate, dstmodel,
		func(length int, h trainer.EvaluateFuncHasher) int {
			var percent, errsum, total atomic.Uint64
			parallel.ForEach(length, 1000, func(j int) {

				var pred uint16

				for i := 0; i < fanout1; i++ {
					var sample = data[j].V2(fanout1, i)
					for jj := 0; jj < sample.Len(); jj++ {
						var io = sample.IO(jj)

						//fmt.Printf("Sample IO %d %d: %v\n", i, jj, sample.IO(jj).SampleSentence.Sample.Sentence)

						var predicted = net.Infer2(io) & 1

						pred += predicted

						if predicted == io.Output() {
							percent.Add(1)
						}
						errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
						total.Add(1)
					}
				}

				h.MustPutUint16(j, pred)
			})
			success := 100 * int(percent.Load()) / int(total.Load())
			println("[success rate]", success, "%", "with", errsum.Load(), "errors")

			return success

		})
	trainer.Resume(net, resume, dstmodel)
	trainer.NewLoopFunc(net, &improved_success_rate, 99, evaluate, trainWorst)()
}
