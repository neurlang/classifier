package datasets

import "sync"

// Tally is used to count votes on dataset features and return the majority votes
type Tally struct {
	// this is for multiway classification layers
	// each input has a map of possible outputs with number of votes
	// the highest vote in the inner map wins
	mapping map[uint16]map[uint64]uint64

	// these are votes in case when the feature caused correct overall result
	// true value is added as +1, false value is voted as -1
	// if the tally is positive we map the feature to true, false if negative
	correct map[uint32]int64

	// these are votes in case when the feature caused better result
	// true value is added as +1, false value is voted as -1
	// if the tally is positive we map the feature to true, false if negative
	improve map[uint32]int64

	mut sync.Mutex

	// isFinalization reports whether we are in the finalization stage
	isFinalization bool

	// improvementPossible reports whether an improvement is possible
	improvementPossible bool
}

// Init initializes the tally dataset structure
func (t *Tally) Init() {
	t.mapping = make(map[uint16]map[uint64]uint64)
	t.correct = make(map[uint32]int64)
	t.improve = make(map[uint32]int64)
}
// Free frees the memory occupied by tally dataset structure
func (t *Tally) Free() {
	t.mapping = nil
	t.correct = nil
	t.improve = nil
}

// SetFinalization sets isFinalization and enables the final stage of training
func (t *Tally) SetFinalization(final bool) {
	t.isFinalization = final
}

// GetImprovementPossible reads improvementPossible
func (t *Tally) GetImprovementPossible() bool {
	return t.improvementPossible
}
// Len estimates the size of tally
func (t *Tally) Len() (o int) {
	t.mut.Lock()
	if len(t.mapping) != 0 {
		o = len(t.mapping)
	} else {
		// we can't do better estimate now
		o = len(t.correct) + len(t.improve)
	}
	t.mut.Unlock()
	return
}
// Improve votes for feature which improved the overall result
func (t *Tally) AddToImprove(feature uint32, vote int8) {
	if vote == 0 {
		return
	}
	t.mut.Lock()
	if !t.isFinalization {
		t.improve[feature] += int64(vote)
		if t.improve[feature] == 0 {
			delete(t.improve, feature)
		}
	}
	t.mut.Unlock()
}

// Correct votes for feature which caused the overall result to be correct
func (t *Tally) AddToCorrect(feature uint32, vote int8, improvement bool) {
	if vote == 0 {
		return
	}
	t.mut.Lock()
	t.correct[feature] += int64(vote)
	if t.correct[feature] == 0 {
		delete(t.correct, feature)
	}
	if improvement {
		t.improvementPossible = true
	}
	t.mut.Unlock()
}

// AddToMapAll adds feature maps to all output votes to mapping
func (t *Tally) AddToMapAll(feature uint16, output uint64, loss func(n uint32) uint32, max uint32) {
	t.mut.Lock()
	if t.mapping[feature] == nil {
		t.mapping[feature] = make(map[uint64]uint64)
	}
	if !t.isFinalization {
		for i := uint32(0); i < max; i++ {
			err := uint64(loss(i))
			t.mapping[feature][uint64(i)] += err * err
		}
	} else {
		t.mapping[feature][output] ++
	}
	t.improvementPossible = true
	t.mut.Unlock()
}

// AddToMap adds feature maps to this output votes to mapping
func (t *Tally) AddToMapping(feature uint16, output uint64) {
	t.mut.Lock()
	if t.mapping[feature] == nil {
		t.mapping[feature] = make(map[uint64]uint64)
	}
	t.mapping[feature][output] ++
	t.improvementPossible = true
	t.mut.Unlock()
}

// Split Splits the tally structure into a splitted dataset
func (t *Tally) Split() SplittedDataset {
	if len(t.mapping) > 0 {
		var mapp Datamap
		mapp.Init()
		for k, freq := range t.mapping {
			var maxk uint64
			for k2 := range freq {
				maxk = k2
				break
			}
			for k2 := range freq {
				if freq[k2] > freq[maxk] {
					maxk = k2
				}
			}
			mapp[k] = maxk
		}
		return mapp.Split()
	} else {
		var sett Dataset
		sett.Init()
		// we initialize the set with pairs which improve first
		for value, rating := range t.improve {
			if rating != 0 {
				sett[value] = rating > 0
			}
		}
		// finally we overwrite the set with pairs which make it correct
		for value, rating := range t.correct {
			if rating != 0 {
				sett[value] = rating > 0
			}
		}
		return sett.Split()
	}
}
