package main

import "sync/atomic"

//import "fmt"
//import "runtime"
//import "math"
//import "math/rand"
import "flag"

//import "os"
import "github.com/neurlang/classifier/datasets/squareroot"

//import "github.com/neurlang/classifier/datasets"
//import "github.com/neurlang/classifier/learning"
//import "github.com/neurlang/classifier/layer/conv2d"
import "github.com/neurlang/classifier/layer/majpool2d"

//import "github.com/neurlang/classifier/layer/full"
//import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/parallel"

func error_abs(a, b uint32) uint32 {
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

	dataset := squareroot.Medium()

	const fanout1 = 3
	const fanout2 = 13
	const fanout3 = 3
	const fanout4 = 13
	//const fanout5 = 3
	//const fanout6 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<fanout6)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
	net.NewLayer(1, squareroot.MediumClasses)

	evaluate := func() {
		var percent, errsum atomic.Uint64
		parallel.ForEach(len(dataset), 1000, func(j int) {
			{
				var io = squareroot.Sample(dataset[j])

				var predicted = net.Infer2(&io)

				if predicted == io.Output()%net.GetClasses() {
					percent.Add(1)
				}
				errsum.Add(uint64(error_abs(uint32(predicted), uint32(io.Output()))))
			}
		})
		success := 100 * int(percent.Load()) / len(dataset)
		println("[infer success rate]", success, "%", "with", errsum.Load(), "errors")

	}

	if resume != nil && *resume && dstmodel != nil {
		net.ReadZlibWeightsFromFile(*dstmodel)
	}
	evaluate()
}
