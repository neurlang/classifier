package layer

// Layer is the layer which can be used for instantiating a combiner
type Layer interface {

	// Lay creates a combiner
	Lay() Combiner
}

