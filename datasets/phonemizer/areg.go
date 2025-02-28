package phonemizer

import "github.com/neurlang/classifier/hash"
import (
	"fmt"
	"sort"
)

type AregSample struct {
	Src string
	Dst string
	Accept bool
}

func (s *AregSample) Feature(n int) uint32 {
	if n & 1 == 1 {
		return hash.StringHash(hash.StringHash(uint32(n), s.Dst), s.Src)
	}
	return hash.StringHash(hash.StringHash(uint32(n), s.Src), s.Dst)
}

func (s *AregSample) Parity() uint16 {
	return 0
}
func (s *AregSample) Output() uint16 {
	if s.Accept {
		return 1
	}
	return 0
}


func (n *AregSample) Key() [2]string {
	return [2]string{n.Dst, n.Src}
}

func NewHistogram(filename string, reverse bool) []string {
	var m = make(map[rune]int)
	loop(filename, func(src, dst string) {
		if reverse {
			src, dst = dst, src
		}
		for _, v := range []rune(dst) {
			m[v]++
		}
	})
	// Step 1: Extract the keys (runes) from the map
	keys := make([]rune, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	// Step 2: Sort the keys based on the corresponding values in the map
	sort.Slice(keys, func(i, j int) bool {
		return m[keys[i]] > m[keys[j]]
	})

	// Step 3: Convert the sorted runes to a slice of strings (if needed)
	sortedStrings := make([]string, len(keys))
	for i, k := range keys {
		sortedStrings[i] = string(k)
	}
	return sortedStrings
}

func NewDatasetAreg(filename, otherfilename string, reverse bool, histogram []string) (out map[[2]string]*AregSample) {
	out = make(map[[2]string]*AregSample)
	loop(otherfilename, func(src, dst string) {
		if reverse {
			src, dst = dst, src
		}
		for i := 0; i < 32; i++ {
			s := &AregSample{
				Src: src,
				Dst: fmt.Sprint(i),
				Accept: false,
			}
			//fmt.Println(s.Src, s.Dst, s.Accept)
			out[s.Key()] = s
		}
	})
	loop(filename, func(src, dst string) {
		if reverse {
			src, dst = dst, src
		}
		for i := 0; i < 32; i++ {
			s := &AregSample{
				Src: src,
				Dst: fmt.Sprint(i),
				Accept: true,
			}
			//fmt.Println(s.Src, s.Dst, s.Accept)
			out[s.Key()] = s
		}
		for _, k := range histogram {
			s := &AregSample{
				Src: src,
				Dst: dst + k,
				Accept: false,
			}
			//fmt.Println(s.Src, s.Dst, s.Accept)
			out[s.Key()] = s
		}
		for i, option := range []rune(dst) {
			s := &AregSample{
				Src: src,
				Dst: string([]rune(dst)[:i+1]),
				Accept: true,
			}
			//fmt.Println(s.Src, s.Dst, s.Accept)
			out[s.Key()] = s
			for _, k := range histogram {
				if k == string(option) {
					break
				}
				s := &AregSample{
					Src: src,
					Dst: string([]rune(dst)[:i]) + string(k),
					Accept: false,
				}
				//fmt.Println(s.Src, s.Dst, s.Accept)
				out[s.Key()] = s
			}
		}
	})
	return
}

func SplitAreg(out map[[2]string]*AregSample) (data []AregSample) {
	for _, v := range out {
		data = append(data, *v)
	}
	return
}	
