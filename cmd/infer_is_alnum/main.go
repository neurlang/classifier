package main

import "sync/atomic"
//import "fmt"
//import "runtime"
//import "math"
//import "math/rand"
import "flag"

//import "os"
import "github.com/neurlang/classifier/datasets/isalnum"
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

	dataset := isalnum.Dataslice{}


	var net feedforward.FeedforwardNetwork
	net.NewLayer(1, 0)


	evaluate := func() {
		var percent, errsum atomic.Uint64
		parallel.ForEach(dataset.Len(), 1000, func(j int) {
			{
				var io = dataset.Get(j)

				var predicted = net.Infer2(&io) % classes
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
