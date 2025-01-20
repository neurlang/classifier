package feedforward

import rand "math/rand"
//import "github.com/neurlang/classifier/hash"

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
				ii = rand.Intn(len(f.layers[i]))
			}
		outer2:
			// try the algorithm a few times, it's better to try and fail than to force an infinite loop
			for try := 0; try < 10; try++ {

				var combiner = f.combiners[i+1].Lay()

				combiner.Put(ii, true)
				
				var position, count int
				for j := 0; j < len(f.layers[i+2]); j++ {
					if combiner.Feature(j) == 0 {
						continue
					}
					position = j
					count++
					if count > 1 {
						break
					}
				}
				if count == 1 {
					q := position
					// Select a that neuron index in the current layer
					o = append(o, base+ii)
					//println(i, ii)
					ii = q
					break outer2
				}
				
				
				jjj := rand.Perm(len(f.layers[i]))

				for _, jj := range jjj {
					if ii == jj {continue;}

					combiner.Put(jj, true)
					
					var words []int
					for j := 0; j < len(f.layers[i+2]); j++ {
						if combiner.Feature(j) == 0 {
							continue
						}
						words = append(words, j)
					}
					
					//for j := 0; j < len(f.layers[i+2]); j++ {
					
					rand.Shuffle(len(words), func(i, j int) {
						words[i], words[j] = words[j], words[i]
					})
					
					
					for _, q := range words {
					
						f1 := combiner.Feature(q)

						combiner.Put(ii, false)
						
						f2 := combiner.Feature(q)

						if  f1 & f2 == f2 && f1 | f2 == f1 {
							// Select a that neuron index in the current layer
							o = append(o, base+ii)
							//println(i, ii)
							ii = q
							break outer2
						}

						combiner.Put(ii, true)

					}
					
					//}

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
