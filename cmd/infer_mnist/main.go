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

	if err := mnist.Error(); err != nil {
		panic(err.Error())
	}

	const fanout1 = 3
	const fanout2 = 5
	const fanout3 = 3
	const fanout4 = 5
	//const fanout5 = 3
	//const fanout6 = 5

	var net feedforward.FeedforwardNetwork
	//net.NewLayerP(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<fanout6)
	//net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
	net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
	net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
	net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
	net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
	net.NewLayer(1, 0)

	evaluate := func() {
		var percent int
		var errsum uint64
		for j := range mnist.InferLabels {
			{
				var input = mnist.SmallInput(mnist.SmallInferSet[j])
				var output = feedforward.SingleValue(mnist.InferLabels[j] & 1)

				var predicted = net.Infer(&input).Feature(0)
				if predicted == output.Feature(0) {
					percent++
				}
				errsum += uint64(error_abs(predicted, output.Feature(0)))
			}
		}
		success := percent * 100 / len(mnist.InferLabels)
		println("[infer success rate]", success, "%", "with", errsum, "errors")

	}
	evaluate2 := func() {
		var percent int
		var errsum uint64
		for j := range mnist.TrainLabels {
			{
				var input = mnist.SmallInput(mnist.SmallTrainSet[j])
				var output = feedforward.SingleValue(mnist.TrainLabels[j] & 1)

				var predicted = net.Infer(&input).Feature(0)
				if predicted == output.Feature(0) {
					percent++
				}
				errsum += uint64(error_abs(predicted, output.Feature(0)))
			}
		}
		success := percent * 100 / len(mnist.TrainLabels)
		println("[train success rate]", success, "%", "with", errsum, "errors")

	}
	if resume != nil && *resume && dstmodel != nil {
		net.ReadCompressedWeightsFromFile(*dstmodel)
	}
	evaluate()
	evaluate2()
}
