package avx

import (
	"log"
	"os"
)

import "github.com/neurlang/classifier/learning"

// SetLogger sets the output logger file where hashtron golang code programs are written
func (h *HyperParameters) SetLogger(filename string) {
	outfile, _ := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	h.l = log.New(outfile, "", 0)
}

type HyperParameters struct {
	learning.HyperParameters

	AvxLanes uint32 // should be set to 16 for AVX512
	AvxSkip  uint32 // should be set to 1 to not skip work, >1 to skip some possibly repeated work

	l *log.Logger
}

func (h *HyperParameters) H() *learning.HyperParameters {
	return &h.HyperParameters
}
