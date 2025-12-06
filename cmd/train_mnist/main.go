package main

import "sync/atomic"

//import "math"
//import "math/rand"
import "flag"
import "github.com/neurlang/classifier/datasets/mnist"
import "github.com/neurlang/classifier/datasets"

//import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"

//import "github.com/neurlang/classifier/layer/sochastic"
//import "github.com/neurlang/classifier/layer/sum"
import "github.com/neurlang/classifier/layer/full"

//import "github.com/neurlang/classifier/hashtron"
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

	trainWorst := trainer.NewTrainWorstFunc(net, nil, nil, nil,
		func(worst []int, tally datasets.AnyTally) {
			parallel.ForEach(len(dataslice), 1000, func(jjj int) {
				var io = dataslice[jjj]
				net.AnyTally(&io, worst, tally, func(actual uint32, expected uint32, mask uint32) uint32 {
					if actual%classes == expected%classes {
						return 0
					}
					return 1
				})
			})
		})
	evaluate := trainer.NewEvaluateFunc(net, len(dataslice), 99, &improved_success_rate, dstmodel,
		func(length int, h trainer.EvaluateFuncHasher) int {
			var percent, errsum atomic.Uint64
			parallel.ForEach(length, 1000, func(j int) {
				var io = dataslice[j]
				var predicted = net.Infer2(&io) % classes

				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			})
			success := 100 * int(percent.Load()) / length
			println("[success rate]", success, "%", "with", errsum.Load(), "errors")
			return success
		})
	trainer.Resume(net, resume, dstmodel)
	if resume != nil && *resume && srcmodel != nil {
		net.ReadZlibWeightsFromFile(*srcmodel)
	}
	trainer.NewLoopFunc(net, &improved_success_rate, 100, evaluate, trainWorst)()

}
