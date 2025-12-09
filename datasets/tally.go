package datasets

import "sync"
import "crypto/rand"
import "encoding/binary"

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

	// global premodulo and salt
	globalPremodulo, globalSalt uint32
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

func (t *Tally) IsGlobalPremodulo() bool {
	return t.globalPremodulo != 0
}
func (t *Tally) SetGlobalPremodulo(mod uint32) {
	var b [4]byte
	rand.Read(b[:])
	t.globalSalt = binary.LittleEndian.Uint32(b[:])
	t.globalPremodulo = mod
}
func (t *Tally) GetGlobalSaltPremodulo() [2]uint32 {
	return [2]uint32{t.globalSalt, t.globalPremodulo}
}
func (t *Tally) GetGlobalPremodulo() uint32 {
	return t.globalPremodulo
}

// SetFinalization sets isFinalization and enables the final stage of training
func (t *Tally) SetFinalization(final bool) {
	t.isFinalization = final
}

// GetImprovementPossible reads improvementPossible
func (t *Tally) GetImprovementPossible() bool {
	t.mut.Lock()
	defer t.mut.Unlock()
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
		t.improvementPossible = true
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
	t.mapping[feature][output]++
	t.improvementPossible = true
	t.mut.Unlock()
}

// AddToMap adds feature maps to this output votes to mapping
func (t *Tally) AddToMapping(feature uint16, output uint64) {
	t.mut.Lock()
	if t.mapping[feature] == nil {
		t.mapping[feature] = make(map[uint64]uint64)
	}
	t.mapping[feature][output]++
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

func (t *Tally) DatasetAt(n int) Dataset {
	return t.Dataset()
}

// Dataset gets binary Dataset from tally
func (t *Tally) Dataset() Dataset {
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
	return sett
}

// GetCellDecision returns the distilled output for a specific cell and feature
func (t *Tally) GetCellDecision(position int, feature uint32) (val bool, avail bool) {
	if position > 0 {
		return
	}
	t.mut.Lock()
	if t.improve[feature] != 0 {
		v, avail := t.improve[feature]
		t.mut.Unlock()
		return v > 0, avail
	} else {
		v, avail := t.correct[feature]
		t.mut.Unlock()
		return v > 0, avail
	}
}

// SetCellDecision stores a distilled decision for a specific cell and feature
func (t *Tally) SetCellDecision(position int, feature uint32, output bool) {
	if position > 0 {
		return
	}
	t.mut.Lock()
	if output {
		t.correct[feature] = 1
		t.improve[feature] = 1
	} else {
		t.correct[feature] = -1
		t.improve[feature] = -1
	}
	t.mut.Unlock()
}
