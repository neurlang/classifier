package main

import "sync/atomic"

import "flag"
import "github.com/neurlang/classifier/datasets/isvirus"
import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/layer/majpool2d"
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
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	dataset := isvirus.Dataslice{}

	const fanout1 = 1
	const fanout2 = 15
	const fanout3 = 1
	const fanout4 = 15
	//const fanout5 = 3
	//const fanout6 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<fanout6)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew2(fanout2, 1, fanout1, 1, fanout2, 1, 1, 0))
	net.NewLayer(1, 0)

	trainWorst := trainer.NewTrainWorstFunc(net, nil, nil, nil,
		func(worst []int, tally datasets.AnyTally) {
			parallel.ForEach(dataset.Len(), 1000, func(jjj int) {
				var io = dataset.Get(jjj).Balance()
				net.AnyTally(&io, worst, tally, nil)
			})
		})
	evaluate := trainer.NewEvaluateFunc(net, dataset.Len(), 99, &improved_success_rate, dstmodel,
		func(length int, h trainer.EvaluateFuncHasher) int {
			var percent, errsum atomic.Uint64
			parallel.ForEach(length, 1000, func(j int) {
				var io = dataset.Get(j).Balance()
				var predicted = net.Infer2(&io)

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
	trainer.NewLoopFunc(net, &improved_success_rate, 100, evaluate, trainWorst)()

}
