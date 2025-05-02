package speak

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Sample struct {
	Candidate uint32
	Source    []rune
	Target    []uint32
	Dim       int
	Incorrect *map[uint32]struct{}
}

func (s *Sample) Len() int {
	if _, ok := (*s.Incorrect)[s.Target[len(s.Target)-1]]; ok {
		return len(*s.Incorrect)
	}
	return len(*s.Incorrect) + 1
	//return int(s.Target[len(s.Target)-1]) + 1
}

func (s *Sample) Alls() (ret []uint32) {
	for k := range *s.Incorrect {
		if k == s.Target[len(s.Target)-1] {
			continue
		}
		ret = append(ret, k)
	}
	ret = append(ret, s.Target[len(s.Target)-1])
	return
}

func (s *Sample) SetOutput(n uint32) {
	s.Candidate = n
}

// Feature: calculates query, keyvalue input for attention matrix
func (s *Sample) Feature(n int) (ret uint32) {
	pos := (n / 2) % (s.Dim / 2)
	if s.Parity() == 1 {
		ret = 1 << 31
	}
	if n%2 == 0 {
		for ; pos < len(s.Source); pos += (s.Dim / 2) {
			ret += ^uint32(s.Source[pos])
		}
		return

	}
	for ; pos < len(s.Target); pos += (s.Dim / 2) {
		if pos+1 == len(s.Target) {
			ret += ^uint32(s.Candidate)
		} else {
			ret += ^uint32(s.Target[pos])
		}
	}
	return
}

func (s *Sample) Parity() (ret uint16) {
	//return 0
	return uint16((len(s.Source)) & 1)
}
func (s *Sample) Output() (ret uint16) {
	// last Target is correct token
	if s.Candidate == s.Target[len(s.Target)-1] {
		return 1
	}
	return 0
}

type Sample2 struct {
	Sample
}

func (s *Sample2) Len() int {
	return s.Sample.Len()
}

func (s *Sample2) Alls() (ret []uint32) {
	return s.Sample.Alls()
}

func (s *Sample2) SetOutput(n uint32) {
	s.Sample.SetOutput(n)
}

func (s *Sample2) Feature(n int) (ret uint32) {
	return s.Sample.Feature(n)
}

func (s *Sample2) Parity() uint16 {
	return 0
}

func (s *Sample2) Output() uint16 {
	// last Target is correct token
	if s.Sample.Candidate <= s.Sample.Target[len(s.Sample.Target)-1] {
		return 1
	}
	return 0
}

func loop(filename string, do func(string, string)) {
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

		// Example: Print the columns
		do(column1, column2)

	}

	// Check for any scanner errors
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}
}

func NewDataset(file string, dimension int) (ret []Sample) {
	var sharedMaps = make(map[int]*map[uint32]struct{})
	loop(file, func(src, dst string) {
		dst_toks := strings.Split(dst, " ")
		var dst_uints = make([]uint32, (len(dst_toks)+2)/3, (len(dst_toks)+2)/3)
		src_toks := []rune(src)
		for i, tok := range dst_toks {
			n, _ := strconv.Atoi(tok)
			shift := (2 - (i % 3)) * 10 // Reverse the shift direction
			dst_uints[i/3] |= uint32(n) << shift
		}
		for i, n := range dst_uints {
			var sharedmapkey int
			if i == 0 {
				sharedmapkey = -1
			} else {
				sharedmapkey = int(dst_uints[i-1])
			}

			if sharedMaps[sharedmapkey] == nil {
				var mapping = make(map[uint32]struct{})
				sharedMaps[sharedmapkey] = &mapping
			}
			(*sharedMaps[sharedmapkey])[n] = struct{}{}
			s := Sample{
				Candidate: n,
				Source:    src_toks,
				Target:    dst_uints[:i+1],
				Dim:       dimension,
				Incorrect: sharedMaps[sharedmapkey],
			}
			ret = append(ret, s)
		}
	})
	return
}

func NewDataset2(file string, dimension int) (ret []Sample2) {
	var sharedMaps = make(map[int]*map[uint32]struct{})
	loop(file, func(src, dst string) {
		dst_toks := strings.Split(dst, " ")
		var dst_uints = make([]uint32, (len(dst_toks)+1)/2, (len(dst_toks)+1)/2)
		src_toks := []rune(src)
		for i, tok := range dst_toks {
			n, _ := strconv.Atoi(tok)
			shift := (1 - (i % 2)) * 15 // Reverse the shift direction
			dst_uints[i/2] |= uint32(n+1) << shift
		}
		for i, n := range dst_uints {
			var sharedmapkey int
			if i == 0 {
				sharedmapkey = -int(src_toks[0])
			} else {
				sharedmapkey = int(dst_uints[i-1])
			}

			if sharedMaps[sharedmapkey] == nil {
				var mapping = make(map[uint32]struct{})
				sharedMaps[sharedmapkey] = &mapping
			}
			(*sharedMaps[sharedmapkey])[n] = struct{}{}
			s := Sample2{
				Sample{
					Candidate: n,
					Source:    src_toks,
					Target:    dst_uints[:i+1],
					Dim:       dimension,
					Incorrect: sharedMaps[sharedmapkey],
				},
			}
			ret = append(ret, s)
		}
	})
	return
}
