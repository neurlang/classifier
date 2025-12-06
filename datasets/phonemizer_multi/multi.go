package phonemizer_multi

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/neurlang/classifier/hash"

	"github.com/yousifnimah/NumToWordsGo/NumToWords"
)

// Sample is one sentence
type Sample struct {
	Sentence []Token
}

type Token struct {
	// homograph = hash of written word == query
	Homograph uint32
	// solution = hash of ipa word == value
	Solution uint32
	// here the fisrt integer is like solution (hash of ipa word), the second is the tag key
	Choices [][2]uint32
}

func (t *Token) Len() int {
	return len(t.Choices)
}

func (s *Sample) V1(dim, pos int) SampleSentence {
	return SampleSentence{
		Sample:    s,
		position:  pos,
		dimension: dim,
		version:   1,
	}
}
func (s *Sample) V2(dim, pos int) SampleSentence {
	return SampleSentence{
		Sample:    s,
		position:  pos,
		dimension: dim,
		version:   2,
	}
}

type SampleSentence struct {
	Sample    *Sample
	position  int
	dimension int
	version   byte
}

func (s *SampleSentence) Len() int {
	if len(s.Sample.Sentence) > s.position {
		return s.Sample.Sentence[s.position].Len()
	}
	return 0
}

type SampleSentenceIO struct {
	SampleSentence *SampleSentence
	choice         int
}

func (s *SampleSentence) IO(n int) (ret *SampleSentenceIO) {
	return &SampleSentenceIO{
		SampleSentence: s,
		choice:         n,
	}
}

// Feature: calculates query, key, value input for attention matrix
// n - if dividible by 3, it's supposed to return the homograph
// n - if equal to 1 divided by 3, it calculates the key token
// n - if equal to 2 divided by 3, it calculates the value token
func (s *SampleSentenceIO) Feature(n int) (ret uint32) {
	pos := (n / 3) % (s.SampleSentence.dimension / 3)
	if s.Parity() == 1 {
		ret = 1 << 31
	}
	if n%3 == 0 {
		for ; pos < len((s.SampleSentence.Sample.Sentence)); pos += (s.SampleSentence.dimension / 3) {
			ret += uint32(s.SampleSentence.Sample.Sentence[pos].Homograph)
		}
		return

	}
	for ; pos < len((s.SampleSentence.Sample.Sentence)); pos += (s.SampleSentence.dimension / 3) {
		if pos < s.SampleSentence.position {
			ret += uint32(s.SampleSentence.Sample.Sentence[pos].Solution)
		} else if pos == s.SampleSentence.position {
			choice := s.SampleSentence.Sample.Sentence[pos].Choices[s.choice]
			// Compare current choice with context
			if n%3 == 1 {
				ret += uint32(choice[1]) // Key
			} else if n%3 == 2 {
				ret += uint32(choice[0]) // Value
			}
		} else if s.SampleSentence.version >= 2 {
			for _, choice := range s.SampleSentence.Sample.Sentence[pos].Choices {
				// Compare future shifted choice with context
				if n%3 == 1 {
					ret += uint32(choice[1]) >> 16 // Key
				} else if n%3 == 2 {
					ret += uint32(choice[0]) >> 16 // Value
				}
			}
		}
	}
	return
}

func (s *SampleSentenceIO) Parity() (ret uint16) {
	//return 0
	return uint16(len(s.SampleSentence.Sample.Sentence) & 1)
}
func (s *SampleSentenceIO) Output() (ret uint16) {
	if s.SampleSentence.Sample.Sentence[s.SampleSentence.position].Choices[s.choice][0] == s.SampleSentence.Sample.Sentence[s.SampleSentence.position].Solution {
		return 1
	}
	return 0
}

func loop(filename string, do func(string, string, string)) {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Create a new scanner to read the file line by line
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		columns := strings.Split(line, "\t")

		// Check if we have exactly two columns
		if len(columns) != 2 && len(columns) != 3 {
			fmt.Println("Line does not have exactly two or three columns:", line)
			continue
		}

		// Process each column
		column1 := columns[0]
		column2 := columns[1]
		var column3 string
		if len(columns) > 2 {
			column3 = columns[2]
		}

		// Example: Print the columns
		do(column1, column2, column3)

	}

	// Check for any scanner errors
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}
}

func addTags(bag map[uint32]string, tags ...string) map[uint32]string {
	for _, v := range tags {
		bag[hash.StringHash(0, v)] = v
	}
	return bag
}

func parseTags(cell string) (ret map[uint32]string) {
	ret = make(map[uint32]string)
	if cell == "" {
		return
	}
	var tags []string
	err := json.Unmarshal([]byte(cell), &tags)
	if err != nil {
		fmt.Printf("Cell tag: %s, Error: %v\n", cell, err)
	}
	for _, v := range tags {
		ret[hash.StringHash(0, v)] = v
	}
	return
}

func serializeTags(tags map[uint32]string) (key uint32, ret string) {
	var tagstrings = []string{}
	for k, v := range tags {
		key ^= k
		tagstrings = append(tagstrings, v)
	}
	sort.Strings(tagstrings)
	data, _ := json.Marshal(tagstrings)
	if len(data) > 0 {
		ret = string(data)
	} else {
		ret = "[]"
	}
	if key == 0 {
		key++
	}
	return
}

func isAllDigits(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, c := range s {
		if !unicode.IsDigit(c) {
			return false
		}
	}
	return true
}

func NewDataset(dir string) (ret []Sample) {

	var tags = make(map[uint32]string)
	var m = make(map[string]map[string]uint32)

	lexiconRowHandler := func(src string, dst, tag string) {
		if _, ok := m[src]; !ok {
			m[src] = make(map[string]uint32)
		}
		var tagstr = "[]"
		if tag != "" {
			tagstr = tag
		}
		if _, ok := m[src][dst]; !ok {
			var tagkey, tagjson = serializeTags(addTags(parseTags(tagstr), "dict"))
			m[src][dst] = tagkey
			tags[tagkey] = tagjson
		} else {
			existingTags := parseTags(tags[m[src][dst]])
			var existing []string
			for _, tag := range existingTags {
				existing = append(existing, tag)
			}
			var tagkey, tagjson = serializeTags(addTags(parseTags(tagstr), existing...))
			m[src][dst] = tagkey
			tags[tagkey] = tagjson
		}
	}

	loop(dir+string(os.PathSeparator)+"lexicon.tsv", lexiconRowHandler)
	loop(dir+string(os.PathSeparator)+"abbr.tsv", lexiconRowHandler)
	// lexicon consistency verification
	for k, v := range m {
		var set = make(map[uint32]string)
		for q, w := range v {
			if qq, ok := set[w]; ok {
				fmt.Println("ERROR: tag collision for '" + k + "': '" + qq + "' vs '" + q + "'")
			}
			set[w] = q
		}
	}

	var is_english = strings.Contains(dir, "english")
	///file, _ := os.Create("/tmp/dump.txt")
	loop(dir+string(os.PathSeparator)+"multi.tsv", func(src string, dst, _ string) {
		srcv := strings.Split(src, " ")
		dstv := strings.Split(dst, " ")
		if len(srcv) != len(dstv) {
			fmt.Println("Line does not have equal number of words:", src, dst)
			return
		}
		var s Sample
		for i := 0; i < len(srcv); i++ {

			if is_english {
				var is_numeric = isAllDigits(srcv[i])
				if is_numeric {
					num, err1 := strconv.Atoi(srcv[i])
					if err1 == nil {
						sentence, err2 := NumToWords.Convert(num, "en")
						if err2 == nil {
							fields := strings.Fields(sentence)
							srcv = slices.Delete(srcv, i, i+1)
							dstv = slices.Delete(dstv, i, i+1)
							srcv = slices.Insert(srcv, i, fields...)
							for range fields {
								dstv = slices.Insert(dstv, i, "_")
							}
						}
					}
				}
			}
		}
		for i := range srcv {
			if srcv[i] == "" && dstv[i] == "" {
				continue
			}
			var one = srcv[i] == "_" || dstv[i] == "_"
			if !one {
				//println("LEXICON:", srcv[i], dstv[i])
			}
			if len(m[srcv[i]]) == 0 {
				fmt.Println("ERROR: Word not in dict:", srcv[i], dstv[i])
				return
			}
			if len(m[srcv[i]]) != 1 && one {
				fmt.Println("ERROR: Word does not have one spoken form:", srcv[i], dstv[i])
				for k, v := range m[srcv[i]] {
					println(k, v, tags[v])
				}
				println()
			}
			// 2) Check that goldIPA matches one of the lexicon keys
			found := false
			for spokenForm := range m[srcv[i]] {
				if spokenForm == dstv[i] {
					found = true
					break
				}
			}
			if !found && dstv[i] != "_" {
				fmt.Println("ERROR: gold pronunciation not in choices for", srcv[i], dstv[i])
				return
			}
			var strkey [][2]string
			for k, v := range m[srcv[i]] {
				strkey = append(strkey, [2]string{k, fmt.Sprint(v)})
			}
			sort.SliceStable(strkey, func(i, j int) bool {
				return strkey[i][0] < strkey[j][0]
			})
			var array [][2]uint32
			for _, v := range strkey {
				num, _ := strconv.Atoi(v[1])
				array = append(array, [2]uint32{hash.StringHash(0, v[0]), uint32(num)})
			}
			sort.SliceStable(array, func(i, j int) bool {
				return array[i][0] < array[j][0]
			})
			var sol uint32
			if dstv[i] == "_" {
				sol = array[0][0]
			} else {
				sol = hash.StringHash(0, dstv[i])
			}
			t := Token{
				Homograph: hash.StringHash(0, srcv[i]),
				Solution:  sol,
				Choices:   array,
			}
			s.Sentence = append(s.Sentence, t)
		}
		///fmt.Fprintln(file, src, s)
		ret = append(ret, s)
	})
	///file.Close()
	return
}
