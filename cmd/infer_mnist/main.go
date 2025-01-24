package main

import "sync/atomic"
//import "fmt"
//import "runtime"
//import "math"
//import "math/rand"
import "flag"

//import "os"
import "github.com/neurlang/classifier/datasets/mnist"

//import "github.com/neurlang/classifier/datasets"
//import "github.com/neurlang/classifier/learning"
//import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"
//import "github.com/neurlang/classifier/layer/sochastic"
//import "github.com/neurlang/classifier/layer/sum"
import "github.com/neurlang/classifier/layer/full"
//import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/parallel"

func error_abs(a, b uint16) uint16 {
	if a > b {
		return a - b
	}
	return b - a
}

func main() {

	dstmodel := flag.String("dstmodel", "", "model destination .json.lzw file")
	flag.Bool("pgo", false, "enable pgo")
	resume := flag.Bool("resume", false, "resume training")
	flag.Parse()
	
	const classes = 10

	trainslice, testslice, _, _, err := mnist.New()
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


	evaluate := func() {
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(testslice), 1000, func(j int) {
			{
				var io = testslice[j]

				var predicted = net.Infer2(&io) % classes
				//println(predicted, io.Output())
				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(predicted, io.Output())))
			}
		})
		success := int(percent.Load()) * 100 / len(testslice)
		println("[infer success rate]", success, "%", "with", errsum.Load(), "errors")

	}
	evaluate2 := func() {
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(trainslice), 1000, func(j int) {
			{
				var io = trainslice[j]

				var predicted = net.Infer2(&io) % classes
				
				//println(predicted, io.Output())
				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(predicted, io.Output())))
			}
		})
		success := int(percent.Load()) * 100 / len(trainslice)
		println("[train success rate]", success, "%", "with", errsum.Load(), "errors")

	}
	if resume != nil && *resume && dstmodel != nil {
		err := net.ReadZlibWeightsFromFile(*dstmodel)
		if err != nil {
			println(err.Error())
		}
	}
	evaluate()
	evaluate2()
}
