// Package layer defines a custom combiner and layer interface
package layer

// Combiner combines input booleans, stores them internally, and combines them to form output features.
type Combiner interface {

	// Put inserts a boolean at position n.
	Put(n int, v bool)

	// Feature returns the n-th feature from the combiner. Next layer reads
	// its inputs using this method for hashtron n in the next layer.
	Feature(n int) (o uint32)

	// Disregard tells whether putting value false at position n would not affect
	// any feature output (as opposed to putting value true at position n).
	// This excludes training samples early which would have no impact,
	// because the next layer would see the same thing regardless of what we put.
	Disregard(n int) bool
}
