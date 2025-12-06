package main

import "sync/atomic"

//import "math"
//import "math/rand"
import "flag"
import "github.com/neurlang/classifier/datasets/squareroot"
import "github.com/neurlang/classifier/datasets"
//import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"
//import "github.com/neurlang/classifier/layer/full"
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
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()

	var improved_success_rate = 0

	dataset := squareroot.Medium()

	const fanout1 = 3
	const fanout2 = 5
	const fanout3 = 1
	const fanout4 = 5

	var net feedforward.FeedforwardNetwork
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4*squareroot.MediumClasses, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*squareroot.MediumClasses*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2*squareroot.MediumClasses, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew2(squareroot.MediumClasses*fanout2, 1, fanout1, 1, squareroot.MediumClasses*fanout2, 1, 1, 0))
	net.NewLayer(squareroot.MediumClasses, 0)

	trainWorst := trainer.NewTrainWorstFunc(net, nil, nil, nil,
		func(worst []int, tally datasets.AnyTally) {
			parallel.ForEach(len(dataset), 1000, func(jjj int) {
				var io = squareroot.Sample(dataset[jjj])
				net.AnyTally(&io, worst, tally, nil)
			})
		})
	evaluate := trainer.NewEvaluateFunc(net, len(dataset), 99, &improved_success_rate, dstmodel,
		func(length int, h trainer.EvaluateFuncHasher) int {
			var percent, errsum atomic.Uint64
			parallel.ForEach(length, 1000, func(j int) {
				var io = squareroot.Sample(dataset[j])
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
