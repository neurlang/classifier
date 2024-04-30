package learning

import (
	"log"
	"os"
)

func (h *HyperParameters) SetLogger(filename string) {
	outfile, _ := os.OpenFile("solutions.txt", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	h.l = log.New(outfile, "", 0)
}

type HyperParameters struct {
	Threads int // number of threads for learning

	Shuffle bool // whether to shuffle the set before each learning attempt
	Seed    bool // seed prng using true rng

	Printer       uint32 // print when hit conflicting solution of at least this large size
	DeadlineMs    int    // deadline in milliseconds to throw away incomplete solution attempt
	DeadlineRetry int    // retry from scratch after this many failed deadlines

	Factor uint32

	Subtractor uint32 // stesub

	InitialLimit int // initial limit of how small the solution must be to be saved to disk

	l *log.Logger
}
