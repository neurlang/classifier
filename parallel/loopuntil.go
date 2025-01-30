// package parallel contains parallel LoopUntil() and parallel ForEach() plus other concurrency primitives.
package parallel

import (
	"math"
	"sync"
	"sync/atomic"
)

// LoopStopper is an interface to check if the loop should stop.
type LoopStopper interface {

	// Load reports true if the loop should stop.
	Load() bool
}

// Loop represents the number of goroutines to run.
type Loop int

// LoopUntil starts 'l' goroutines that iterate until one of them stops the loop.
// Each goroutine processes a unique integer i starting from 0.
// The loop stops if i reaches math.MaxUint32 or any goroutine's yield returns true.
func (l Loop) LoopUntil(yield func(i uint32, ender LoopStopper) bool) {
	var (
		i     uint32              // Atomic counter for the current index.
		ender atomic.Bool         // Atomic boolean to signal stop.
		wg    sync.WaitGroup      // WaitGroup to wait for all goroutines.
	)

	// Start 'l' goroutines.
	for n := 0; n < int(l); n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				// Check if the loop should stop.
				if ender.Load() {
					return
				}

				// Atomically increment i and get the new value.
				newI := atomic.AddUint32(&i, 1)
				if newI == math.MaxUint32 {
					// Special case: Terminate if i reaches MaxUint32.
					ender.Store(true)
					return
				}

				// The current index to process (previous value of i).
				currentI := newI - 1

				// Execute the yield function. If it returns true, stop the loop.
				if yield(currentI, &ender) {
					ender.Store(true)
					return
				}
			}
		}()
	}

	// Wait for all goroutines to finish.
	wg.Wait()
}
