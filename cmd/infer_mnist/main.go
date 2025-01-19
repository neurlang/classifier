package main

//import "sync"
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
//import "github.com/neurlang/classifier/layer/full"
//import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/net/feedforward"

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

	_, _, trainslice, testslice, err := mnist.New()
	if err != nil {
		panic(err.Error())
	}

	const fanout1 = 1
	const fanout2 = 7
	const fanout3 = 1
	const fanout4 = 7

	var net feedforward.FeedforwardNetwork
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew2(fanout2, 1, fanout1, 1, fanout2, 1, 1, 0))
	net.NewLayer(1, 4)


	evaluate := func() {
		var percent int
		var errsum uint64
		for j := range testslice {
			{
				var io = testslice[j]

				var predicted = net.Infer2(&io) % 10
				if predicted == io.Output()%net.GetClasses() {
					percent++
				}
				errsum += uint64(error_abs(predicted, io.Output()))
			}
		}
		success := percent * 100 / len(testslice)
		println("[infer success rate]", success, "%", "with", errsum, "errors")

	}
	evaluate2 := func() {
		var percent int
		var errsum uint64
		for j := range trainslice {
			{
				var io = trainslice[j]

				var predicted = net.Infer2(&io) % 10
				if predicted == io.Output()%net.GetClasses() {
					percent++
				}
				errsum += uint64(error_abs(predicted, io.Output()))
			}
		}
		success := percent * 100 / len(trainslice)
		println("[train success rate]", success, "%", "with", errsum, "errors")

	}
	if resume != nil && *resume && dstmodel != nil {
		net.ReadZlibWeightsFromFile(*dstmodel)
	}
	evaluate()
	evaluate2()
}
