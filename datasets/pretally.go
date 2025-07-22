package datasets

import "sync"
import "crypto/rand"
import "encoding/binary"

// PreTally stores distilled decisions for multiple cells with thread-safe access
type PreTally struct {
	maps  []map[uint32]bool
	mutex sync.RWMutex

	// global premodulo and salt
	globalPremodulo, globalSalt uint32

}


// Init resets pretally to be empty
func (d *PreTally) Init() {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.maps = nil
}


func (t *PreTally) IsGlobalPremodulo() bool {
	return t.globalPremodulo != 0
}
func (t *PreTally) SetGlobalPremodulo(mod uint32) {
	var b [4]byte
	rand.Read(b[:])
	t.globalSalt = binary.LittleEndian.Uint32(b[:])
	t.globalPremodulo = mod
}
func (t *PreTally) GetGlobalSaltPremodulo() [2]uint32 {
	return [2]uint32{t.globalSalt, t.globalPremodulo}
}
func (t *PreTally) GetGlobalPremodulo() uint32 {
	return t.globalPremodulo
}

// GetCellDecision returns the distilled output for a specific cell and feature
func (d *PreTally) GetCellDecision(position int, feature uint32) (bool, bool) {
	if position < 0 {
		return false, false
	}

	d.mutex.RLock()
	defer d.mutex.RUnlock()

	if position >= len(d.maps) {
		return false, false
	}


	val, exists := d.maps[position][feature]
	return val, exists
}

// SetCellDecision stores a distilled decision for a specific cell and feature
func (d *PreTally) SetCellDecision(position int, feature uint32, output bool) {
	if position < 0 {
		return
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()

	for position >= len(d.maps) {
		d.maps = append(d.maps, make(map[uint32]bool))
	}

	d.maps[position][feature] = output
}

func (d *PreTally) Len() (ret int) {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	for _, m := range d.maps {
		ret += len(m)
	}
	return
}
func (d *PreTally) Free() {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.maps = nil
}
func (d *PreTally) DatasetAt(position int) Dataset {
	if position < 0 {
		return nil
	}
	d.mutex.RLock()
	defer d.mutex.RUnlock()
	if position >= len(d.maps) {
		return nil
	}
	return d.maps[position]
}

// GetImprovementPossible reads improvementPossible
func (t *PreTally) GetImprovementPossible() bool {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	for _, m := range t.maps {
		if len(m) > 0 {
			return true
		}
	}
	return false
}
// AddToCorrect votes for feature which caused the overall result to be correct
func (t *PreTally) AddToCorrect(feature uint32, vote int8, improvement bool) {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	if len(t.maps) == 0 {
		t.maps = append(t.maps, make(map[uint32]bool))
	}
	t.maps[0][feature] = vote > 0
}
// AddToImprove votes for feature which caused the overall result to be correct
func (t *PreTally) AddToImprove(feature uint32, vote int8) {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	if len(t.maps) == 0 {
		t.maps = append(t.maps, make(map[uint32]bool))
	}
	t.maps[0][feature] = vote > 0
}
// AddToMapping adds feature maps to this output votes to mapping
func (t *PreTally) AddToMapping(feature uint16, output uint64) {
	// not supported
}
