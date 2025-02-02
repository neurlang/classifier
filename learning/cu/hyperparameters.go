package cu

import (
	"log"
	"os"
)

import "github.com/neurlang/classifier/learning"
import "gorgonia.org/cu"

// SetLogger sets the output logger file where hashtron golang code programs are written
func (h *HyperParameters) SetLogger(filename string) {
	outfile, _ := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	h.l = log.New(outfile, "", 0)
}

type HyperParameters struct {
	learning.HyperParameters

	CuCutoff        uint32 // the switchover point to cuda. Smaller problems go to cuda
	CuMemoryBytes   uint64 // statically set memory
	CuMemoryPortion uint16 // how many percent of gpu memory to use. 2=half, 3=third
	CuErase         bool   // pre-erase memory set


	set, input0, input1, result *cu.DevicePtr

	ctx                *cu.CUContext
	inputNums          *cu.DevicePtr
	fn, fn1, fn2       *cu.Function
	stream             *cu.Stream
	backoff            uint64
	iter               uint32
	setSize            int64

	l *log.Logger
}

func (h *HyperParameters) H() *learning.HyperParameters {
	return &h.HyperParameters
}
