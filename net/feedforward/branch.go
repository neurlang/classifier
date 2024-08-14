package feedforward

import "math/rand"

func (f FeedforwardNetwork) Branch(reverse bool) (o []int) {
	o = make([]int, 0, f.LenLayers())

	// Initialize base to account for the starting index of each layer
	base := 0

	var combiner = f.combiners[1].Lay()

	var ii int

	// Traverse through each layer and select a random parent neuron
	for i := 0; i < f.LenLayers(); i++ {
		if len(f.layers[i]) == 0 {
			continue
		}

		if i == 0 {

			var q int

		outer:
			for {

				q = rand.Intn(len(f.layers[i]))

				combiner.Put(q, true)

				for j := 0; j < len(f.layers[i+2]); j++ {
					if combiner.Feature(j) != 0 {
						ii = j
						break outer
					}
				}

			}

			// Select a that neuron index in the current layer
			o = append(o, base+q)

		} else if i+1 < f.LenLayers() {

			combiner = f.combiners[i+1].Lay()

			combiner.Put(ii, true)

		outer2:
			for jjj := 0; jjj < len(f.layers[i])*len(f.layers[i]); jjj++ {
				jj := rand.Intn(len(f.layers[i]))

				combiner.Put(jj, true)

				for j := 0; j < len(f.layers[i+2]); j++ {

					combiner.Put(ii, combiner.Feature(j) == 0)

					if combiner.Feature(j) == 0 {
						ii = j
						// Select a that neuron index in the current layer
						o = append(o, base+jj)
						break outer2
					}

					combiner.Put(ii, combiner.Feature(j) != 0)

				}

			}

		} else {
			// Select last neuron
			o = append(o, base)
		}
		// Update the base index to the start of the current layer
		base += len(f.layers[i])
	}

	// If reverse is true, reverse the slice
	if reverse {
		for i := len(o)/2 - 1; i >= 0; i-- {
			opp := len(o) - 1 - i
			o[i], o[opp] = o[opp], o[i]
		}
	}

	return o
}
