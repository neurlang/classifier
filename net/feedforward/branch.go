package feedforward

import "math/rand"

func (f FeedforwardNetwork) Branch(reverse bool) (o []int) {
	o = make([]int, 0, f.LenLayers())
	
	// Initialize base to account for the starting index of each layer
	base := 0

	// Traverse through each layer and select a random parent neuron
	for i := 0; i < f.LenLayers(); i++ {
		if len(f.layers[i]) == 0 {
			continue
		}

		
		// Select a random neuron index in the current layer
		o = append(o, base + rand.Intn(len(f.layers[i])))
		
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

