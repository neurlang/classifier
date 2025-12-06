package parallel

import "sync"
import "encoding/binary"

// MoveSet is a thread-safe set that tracks unique moves for a specific level.
// When the level changes, all existing moves are cleared to prevent mixing moves from different levels.
type MoveSet struct {
	mu    sync.RWMutex          // Synchronizes access to the set and level
	set   map[[40]byte]struct{} // Stores serialized moves as keys
	level byte                  // Current level associated with the moves in the set
}

// NewMoveSet initializes and returns a new MoveSet instance. Initial level is 0.
func NewMoveSet() *MoveSet {
	return &MoveSet{
		set: make(map[[40]byte]struct{}),
	}
}

// Insert adds a move to the set. If the provided level differs from the current level,
// the set is cleared before adding the new move. This ensures moves are grouped by level.
// position: The game position represented as a 32-byte array.
// move: The move identifier to be serialized.
// level: The current level associated with the move.
func (m *MoveSet) Insert(position [32]byte, move int, level byte) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.level == level {
		m.set[serialize(position, move)] = struct{}{}
	} else {
		// Level changed: reset set and update level
		m.set = make(map[[40]byte]struct{})
		m.level = level
	}
}

// Exists checks if a move exists in the set for the current level.
// Returns false immediately if the provided level doesn't match the set's current level.
// position: The game position to check.
// move: The move identifier to check.
// level: The level to verify against.
func (m *MoveSet) Exists(position [32]byte, move int, level byte) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.level != level {
		return false
	}

	_, exists := m.set[serialize(position, move)]
	return exists
}

// serialize combines position and move into a fixed-size 40-byte key.
// Implementations should use consistent byte encoding for the move parameter.
func serialize(position [32]byte, move int) [40]byte {
	var key [40]byte
	copy(key[:32], position[:])

	// Example serialization using little-endian encoding for the move
	binary.LittleEndian.PutUint64(key[32:], uint64(move))
	return key
}
