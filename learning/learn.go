// Package Learning implements the learning stage of the Neurlang classifier
package learning

//import "math/bits"
import "math/rand"
import crypto_rand "crypto/rand"
import "encoding/binary"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hashtron"

type modulo_t = uint32

// Training trains a single hashtron on a dataset d. It outputs the trained hashtron if successful, or an error.
func (h *HyperParameters) Training(d datasets.Splitter) (*hashtron.Hashtron, error) {

	if h.EOL == nil || len(h.EOL) == 0 {
		h.EOL = []byte{';', ' '}
	}

	var sd = d.Split()

	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}

	var backup = h.InitialLimit
	var result *hashtron.Hashtron

	h.InitialLimit, result = h.Solve(sd)
	for !h.EndWhenSolved || result == nil {
		h.InitialLimit, result = h.Solve(sd)
	}
	h.InitialLimit = backup

	return result, nil
}

// Solve directly solves a single hashtron on a splitted dataset d. It outputs the size of solution
// and the trained hashtron if successful. Most callers should use Training instead.
func (h *HyperParameters) Solve(d datasets.SplittedDataset) (int, *hashtron.Hashtron) {

	var bits byte = 0
	var doit = true
	data := [2][]uint32{}
	for j := range data {
		for v := range d[j] {
			data[j] = append(data[j], v)
			if doit {
				possible := v >> 16
				if possible <= 16 {
					possible++
					if bits < byte(possible) {
						bits = byte(possible)
					}
				} else {
					bits = 0
					doit = false
				}
			}
		}
	}
	prog := h.byReducing(&data)
	if len(prog) == 0 {
		tron, err := hashtron.New(nil, 0)
		if err != nil {
			return h.InitialLimit, nil
		}
		return 1, tron
	}
	tron, err := hashtron.New(prog, bits)
	if err != nil {
		println("Error creating hashtron:", err.Error())
		return h.InitialLimit, nil
	}
	return len(prog), tron
}
