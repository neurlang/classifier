package feedforward

import rand "math/rand/v2"

func (f FeedforwardNetwork) Branch(reverse bool) (o []int) {
	o = make([]int, 0, f.LenLayers())

	// Initialize base to account for the starting index of each layer
	base := 0

	var ii int

	// Traverse through each layer and select a random parent neuron
	for i := 0; i < f.LenLayers(); i++ {
		if len(f.layers[i]) == 0 {
			continue
		}

		if i+1 < f.LenLayers() {
			if i == 0 {
				ii = rand.IntN(len(f.layers[i]))
			} else {
				ii %= len(f.layers[i])
			}
		outer2:
			for jjjj := 0; jjjj < len(f.layers[i])*len(f.layers[i]); jjjj++ {

				var combiner = f.combiners[i+1].Lay()

				combiner.Put(ii, true)

				for jjj := 0; jjj < len(f.layers[i])*len(f.layers[i]); jjj++ {
					jj := rand.IntN(len(f.layers[i]))

					combiner.Put(jj, true)

					for j := 0; j < len(f.layers[i+2])*len(f.layers[i+2]); j++ {
						q := rand.IntN(len(f.layers[i+2]))
					
						if combiner.Feature(q) == 0 {
							continue
						}

						combiner.Put(ii, false)

						if combiner.Feature(q) == 0 {
							// Select a that neuron index in the current layer
							o = append(o, base+ii)
							ii = q
							break outer2
						}

						combiner.Put(ii, true)

					}

				}
			}

		} else if !reverse {
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
