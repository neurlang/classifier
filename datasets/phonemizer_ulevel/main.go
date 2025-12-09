package phonemizer_ulevel

//import "fmt"

// Phonetic utterance-level G2P data loader and subsample generator
// - On-the-fly subsample generation
// - Cyclic-add Q/K/V across configurable Slots
// - Lemire-bounded token hashing (hashtron) with PAD=0
// - Counts per-slot (number of negatives dumped for target slot)
// - Label retained

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"

	"github.com/neurlang/classifier/hash"
)

const (
	// FUTURE_SHIFT is the default attenuation for future tokens (as in homograph): 32 bits
	FUTURE_SHIFT = 32
	// Lemire bound range parameter: 2^32 - 2
	lemireRange = uint32(0xFFFFFFFE)
)

// LanguageJSON is the expected structure of language.json
type LanguageJSON struct {
	Map map[string][]string `json:"Map"`
}

// Sample represents one aligned utterance from clean.tsv
// Values holds lookups from language.json: for each src token a slice of candidate spoken forms (priority sorted)
type Sample struct {
	Src    []string
	Dst    []string
	Values [][]string
	Coll   []map[string]struct{}
	Slots  int
}

// Subsample is the conceptual record for a training example
type Subsample struct {
	Q      []uint32 // len == Slots
	K      []uint32
	V      []uint32
	Counts []uint16
	Label  bool
}

// Feature API: n indexes into 3*Slots dims. n%3: 0->Q,1->K,2->V
func (s *Subsample) Feature(n int) uint32 {
	if len(s.Q) == 0 {
		return 0
	}
	pos := (n / 3) % len(s.Q)
	switch n % 3 {
	case 0:
		return s.Q[pos]
	case 1:
		return s.K[pos]
	case 2:
		return s.V[pos]
	}
	return 0
}

func (s *Subsample) Output() uint16 {
	if s.Label {
		return 1
	}
	return 0
}

func (s *Subsample) Parity() (ret uint16) {
	return 0
}

// Loader handles language.json parsing and dataset parameters
type dataset = struct {
	LanguageJSON
}

// NewLoader loads language.json and prepares the loader with given slots
func NewDataset(dir string, reverse bool, slots int) (out []Sample) {
	var rev string
	if reverse {
		rev = "_reverse"
	}
	cleanTSVPath, languageJSONPath := dir+string(os.PathSeparator)+"clean"+rev+".tsv", dir+string(os.PathSeparator)+"language"+rev+".json"
	f, err := os.Open(languageJSONPath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()
	var lj LanguageJSON
	dec := json.NewDecoder(f)
	if err := dec.Decode(&lj); err != nil {
		panic(err.Error())
	}
	if lj.Map == nil {
		lj.Map = make(map[string][]string)
	}
	file, err := os.Open(cleanTSVPath)
	if err != nil {
		panic(err.Error())
	}
	defer file.Close()

	dedup := make(map[[2]string]struct{})
	collisions := make(map[[2]string][][2]string)

	ds := &dataset{lj}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		cols := strings.Split(line, "	")
		if len(cols) < 2 {
			// ignore malformed
			continue
		}
		if _, ok := dedup[[2]string{cols[0], cols[1]}]; ok {
			continue
		}
		dedup[[2]string{cols[0], cols[1]}] = struct{}{}

		src := strings.Fields(cols[0])
		dst := strings.Fields(cols[1])
		if len(src) != len(dst) {
			// skip inconsistent line
			continue
		}

		query := strings.Join(src, " ")
		for i := range dst {
			prefix := strings.Join(dst[:i], " ")
			collisions[[2]string{
				query,
				prefix,
			}] = append(collisions[[2]string{
				query,
				prefix,
			}], [2]string{src[i], dst[i]})
		}

		var s = Sample{Src: src, Dst: dst, Slots: slots}
		s.Values = make([][]string, len(s.Src))
		for i, tok := range s.Src {
			if v, ok := ds.Map[tok]; ok {
				s.Values[i] = v
			} else {
				s.Values[i] = []string{s.Dst[i]}
			}
		}
		out = append(out, s)
	}
	for j := range out {
		query := strings.Join(out[j].Src, " ")
		for i := range out[j].Dst {
			prefix := strings.Join(out[j].Dst[:i], " ")
			if len(collisions[[2]string{
				query,
				prefix,
			}]) > 1 {
				for _, v := range collisions[[2]string{
					query,
					prefix,
				}] {
					if v[0] == out[j].Src[i] && v[1] != out[j].Dst[i] {
						if out[j].Coll == nil {
							out[j].Coll = make([]map[string]struct{}, len(out[j].Src), len(out[j].Src))
						}
						if out[j].Coll[i] == nil {
							out[j].Coll[i] = make(map[string]struct{})
						}
						out[j].Coll[i][v[1]] = struct{}{}
						//println("COLLISION", query, "|", i, "|", prefix, "|", out[j].Src[i], "|", out[j].Dst[i], "|", v[0], "|", v[1])
					}
				}

			}
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err.Error())
	}
	return out
}

// tokenHash normalizes token -> [1..0xFFFFFFFF], PAD is 0
func tokenHash(token string) uint32 {
	// first-level string hash
	h0 := hash.StringHash(0, token)
	// Lemire bounded hash: returns 0..lemireRange
	h1 := hash.Hash(h0, 0, lemireRange)
	// shift +1 -> 1..0xFFFFFFFF
	return h1 + 1
}

// makeSubsampleForCandidate builds a Subsample for target position i and a given candidate string
// sample: Sample utterance; i: target index; candidate: spoken candidate for target; label: true if gold
// slots: number of slots used (e.g., 8)
func makeSubsampleForCandidate(sample *Sample, i int, candidate string, negativesCount int, slots int) *Subsample {
	if i < 0 || i >= len(sample.Src) {
		panic("target index out of range")
	}
	if slots <= 0 {
		slots = 8
	}

	var s Subsample
	// allocate
	s.Q = make([]uint32, slots)
	s.K = make([]uint32, slots)
	s.V = make([]uint32, slots)
	s.Counts = make([]uint16, slots)

	// 1) Q: Join all source tokens, break into runes, hash each rune, and distribute cyclically
	fullSrc := strings.Join(sample.Src, "")
	runes := []rune(fullSrc)
	for j, r := range runes {
		slot := j % slots
		s.Q[slot] = tokenHash(string(r)) // Hash each individual rune
	}

	// 2) K/V accumulation across entire utterance (cyclic-add fold)
	for t := 0; t < len(sample.Src); t++ {
		slot := t % slots
		// orthographic token hash
		h := tokenHash(sample.Src[t])
		// spoken token hash (gold) for V baseline
		vh := tokenHash(sample.Dst[t])
		// if future relative to target, attenuate
		if t > i {
			// FUTURE_SHIFT is 32, which would clear a uint32 completely
			h = 0
			vh = 0
		}
		// accumulate with uint32 wrap-around
		s.K[slot] = s.K[slot] + h
		s.V[slot] = s.V[slot] + vh
	}

	// 3) Override target slot K/V with explicit target (non-shifted)
	targetSlot := i % slots
	// K target = orthographic token hash (non-shifted)
	s.K[targetSlot] = tokenHash(sample.Src[i])
	// V target = candidate hash (non-shifted)
	if candidate == "" {
		panic("empty candidate")
	}
	candidateHash := tokenHash(candidate)
	s.V[targetSlot] = candidateHash

	// 4) Counts: store negativesCount in target slot, others zero
	if negativesCount < 0 {
		negativesCount = 0
	}
	if negativesCount > 0xFFFF {
		negativesCount = 0xFFFF
	}
	if uint16(negativesCount) > s.Counts[targetSlot] {
		s.Counts[targetSlot] = uint16(negativesCount)
	}

	// collision handling
	if len(sample.Coll) > 0 {
		if _, ok := sample.Coll[i][candidate]; ok {
			s.Label = true
		}
	}

	return &s
}

// V1 returns all subsamples for a sample according to default policy
// default: for each position i emit gold subsample (label true) and negatives that are higher-priority than gold
// Uses Sample.Values if available, otherwise relies on provided candidates map
func (sample *Sample) V1() []*Subsample {
	slots := sample.Slots
	var out []*Subsample
	M := len(sample.Src)
	for i := 0; i < M; i++ {
		cands := sample.Values[i]
		// if no candidates in Values, fallback to the gold only
		if cands == nil || len(cands) == 0 {
			cands = []string{sample.Dst[i]}
		}
		// find gold index in cands
		goldIdx := -1
		for idx, v := range cands {
			if v == sample.Dst[i] {
				goldIdx = idx
				break
			}
		}
		// if gold not found, append it at the end and consider no higher-priority negatives
		if goldIdx == -1 {
			cands = append(cands, sample.Dst[i])
			goldIdx = len(cands) - 1
		}

		// count higher-priority negatives
		negCount := goldIdx // number of candidates with index < goldIdx

		// First emit gold subsample (position_inside_subsample == 0)
		goldSub := makeSubsampleForCandidate(sample, i, sample.Dst[i], negCount, slots)
		if goldSub == nil {
			panic("gold sub must exist")
		}
		goldSub.Label = true
		out = append(out, goldSub)

		//if strings.Join(sample.Src, "") == "sledujte" {
		//	fmt.Println(sample.Src, sample.Dst, goldSub.Q, goldSub.K, goldSub.V, goldSub.Label)
		//}

		// Emit higher-priority negatives in order (index 0 .. goldIdx-1)
		for n := 0; n < goldIdx; n++ {
			neg := cands[n]
			sub := makeSubsampleForCandidate(sample, i, neg, negCount, slots)
			if sub != nil {
				out = append(out, sub)

				//if strings.Join(sample.Src, "") == "sledujte" {
				//	fmt.Println(sample.Src, sample.Dst, sub.Q, sub.K, sub.V, sub.Label)
				//}
			}
		}
	}
	return out
}

// NewInferenceSubsample builds a Subsample for inference given
// full src tokens, past dst tokens, and a candidate option.
//
// src: full orthographic tokens of the utterance
// dst: spoken tokens for positions < cur
// option: candidate spoken token for position cur (== len(dst))
// slots: number of slots (e.g. 8)
func NewInferenceSubsample(src []string, dst []string, option string, slots int) *Subsample {
	cur := len(dst) // current index
	if len(src) == 0 {
		panic("NewInferenceSubsample: src cannot be empty")
	}
	if cur >= len(src) {
		panic("NewInferenceSubsample: dst too long (no current position)")
	}
	if option == "" {
		panic("NewInferenceSubsample: option cannot be empty")
	}
	if slots <= 0 {
		slots = 8
	}

	var s Subsample
	s.Q = make([]uint32, slots)
	s.K = make([]uint32, slots)
	s.V = make([]uint32, slots)
	s.Counts = make([]uint16, slots)

	// 1) Q: Join all source tokens, break into runes, hash each rune, and distribute cyclically
	fullSrc := strings.Join(src, "")
	runes := []rune(fullSrc)
	for j, r := range runes {
		slot := j % slots
		s.Q[slot] = tokenHash(string(r)) // Hash each individual rune
	}

	// --- 2) K/V accumulation across utterance
	for t := 0; t < len(src); t++ {
		slot := t % slots
		h := tokenHash(src[t])
		var vh uint32

		if t < cur {
			vh = tokenHash(dst[t])
		} else if t == cur {
			vh = tokenHash(option)
		} else {
			// future token → attenuate (FUTURE_SHIFT is 32, which clears uint32)
			h = 0
			vh = 0 // no option for future at inference
		}

		s.K[slot] += h
		s.V[slot] += vh
	}

	// --- 3) Label = false, Counts = 0
	s.Label = false

	//fmt.Println(src, dst, option, s.Q, s.K, s.V)

	return &s
}
