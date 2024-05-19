package cu

import (
	"log"
	"os"
)

import "gorgonia.org/cu"

// SetLogger sets the output logger file where hashtron golang code programs are written
func (h *HyperParameters) SetLogger(filename string) {
	outfile, _ := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	h.l = log.New(outfile, "", 0)
}

type HyperParameters struct {
	Threads int // number of threads for learning

	Shuffle bool // whether to shuffle the set before each learning attempt
	Seed    bool // seed prng using true rng

	Printer       uint32 // print when hit conflicting solution of at least this large size
	DeadlineMs    int    // deadline in milliseconds to throw away incomplete solution attempt
	DeadlineRetry int    // retry from scratch after this many failed deadlines

	// Factor is how hard to try to come up with a smaller solution (default: 1)
	// Usually set equal to Subtractor
	Factor uint32

	// Subtractor is how hard to try to come up with a smaller solution (default: 1)
	// Usually set equal to Factor
	Subtractor uint32

	InitialLimit int // initial limit of how small the solution must be to be saved to disk

	DisableProgressBar bool // disable progress bar

	EndWhenSolved bool // end when solved

	Name string // override model name
	EOL  []byte // override EOL string

	CuMemoryBytes   uint64 // statically set memory
	CuMemoryPortion uint16 // how many percent of gpu memory to use. 2=half, 3=third

	ctx                *cu.CUContext
	set, input, result *cu.DevicePtr
	fn, fn1, fn2       *cu.Function
	stream             *cu.Stream
	backoff            uint64
	iter               uint32

	l *log.Logger
}
