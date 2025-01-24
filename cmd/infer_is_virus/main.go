package main

import "sync/atomic"
//import "fmt"
//import "runtime"
//import "math"
//import "math/rand"
import "flag"

//import "os"
import "github.com/neurlang/classifier/datasets/isvirus"
import "github.com/neurlang/classifier/layer/majpool2d"
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


	evaluate := func() {
		var percent, errsum atomic.Uint64
		parallel.ForEach(dataset.Len(), 1000, func(j int) {
			{
				var io = dataset.Get(j).Balance()

				var predicted = net.Infer2(&io) & 1
				//println(predicted, io.Output())
				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(predicted, io.Output())))
			}
		})
		success := int(percent.Load()) * 100 / dataset.Len()
		println("[success rate]", success, "%", "with", errsum.Load(), "errors")

	}
	if resume != nil && *resume && dstmodel != nil {
		err := net.ReadZlibWeightsFromFile(*dstmodel)
		if err != nil {
			println(err.Error())
		}
	}
	evaluate()
}
