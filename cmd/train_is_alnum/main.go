package main

import "runtime"

import "github.com/neurlang/classifier/datasets/isalnum"
import "github.com/neurlang/classifier/learning"

func main() {
	var h learning.HyperParameters
	var dataset = isalnum.Dataset

	h.Threads = runtime.NumCPU()
	h.InitialModulo = 4096 + 1 // depends on problem size

	// shuffle before solving attempts
	h.Shuffle = true
	h.Seed = true

	// restart when stuck
	h.DeadlineMs = 1000
	h.DeadlineRetry = 3

	// affects how fast is the modulo reduced (by 20/21 and then by -1)
	h.Numerator = 20
	h.Denominator = 21
	h.Subtractor = 1

	// reduce Backtracking printing on the log
	h.Printer = 70

	// save any solution to disk
	h.InitialLimit = 99999999
	h.SetLogger("solutions.txt")

	h.Training(dataset)
}
