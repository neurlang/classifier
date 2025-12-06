package main

import "sync/atomic"

//import "math"
//import "math/rand"
import "flag"
import "os"
import "github.com/neurlang/classifier/datasets/isalnum"
import "github.com/neurlang/classifier/datasets"
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
	var difficulty uint32 = 1

	dataset := isalnum.Dataslice{}

	// a net with one neuron
	var net feedforward.FeedforwardNetwork
	net.NewLayer(1, 0)
	

	trainWorst := trainer.NewTrainWorstFunc(net, nil, nil, nil,
		func(worst []int, tally datasets.AnyTally) {
			parallel.ForEach(dataset.Len(), 1000, func(jjj int) {
				var io = dataset.Get(jjj)
				net.AnyTally(&io, worst, tally, nil)
			})
		})
	evaluate := trainer.NewEvaluateFunc(net, dataset.Len(), 99, &improved_success_rate, dstmodel,
		func(length int, h trainer.EvaluateFuncHasher) int {
			var percent, errsum atomic.Uint64
			parallel.ForEach(length, 1000, func(j int) {
				var io = dataset.Get(j)
				var predicted = net.Infer2(&io)

				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			})
			success := 100 * int(percent.Load()) / length
			println("[success rate]", success, "%", "with", errsum.Load(), "errors")
			
			if success == 100 {
				if difficulty > 1000 {
					println("Max accuracy or wrong data. Exiting")
					os.Exit(0)
				} else {
					difficulty++
				}
			}
			return success
		})
	trainer.Resume(net, resume, dstmodel)
	trainer.NewLoopFunc(net, &improved_success_rate, 100, evaluate, trainWorst)()

}
