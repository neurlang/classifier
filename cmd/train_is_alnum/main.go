// Train on Is Alnum dataset
package main

import "runtime"

import "github.com/neurlang/classifier/datasets/isalnum"
import "github.com/neurlang/classifier/learning"

func main() {
	var h learning.HyperParameters
	var dataset = isalnum.Dataset

	h.Threads = runtime.NumCPU()
	//affects initial modulo
	h.Factor = 6

	// shuffle before solving attempts
	h.Shuffle = true
	h.Seed = true

	// restart when stuck
	h.DeadlineMs = 1000
	h.DeadlineRetry = 3

	// additional modulo reduction, affects solution size
	h.Subtractor = 6

	// reduce Backtracking printing on the log
	h.Printer = 70

	// save any solution to disk
	h.InitialLimit = 99999999
	h.SetLogger("solutions.txt")

	h.Training(dataset)
}
