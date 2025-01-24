package phonemizer

import "github.com/neurlang/classifier/hash"
import (
	"bufio"
	"fmt"
	"os"
	"strings"
	//"encoding/json"
)

type NewSample struct {
	SrcA   []string
	DstA   []string
	SrcCut []string
	SrcFut []string
	Option string
	Out bool
	Len int
	I int
	J int
}

func (s *NewSample) Feature(n int) uint32 {
	if n == 0 {
		return hash.StringHash(0, s.Option) ^ hash.StringsHash(0, s.SrcA)
	}
	srcFut := []string(s.SrcFut)
	srcFut = append(srcFut, s.Option)
	srcCut := []string(s.SrcCut)
	srcCut = append(srcCut, s.Option)
	dstA := []string(s.DstA)
	dstA = append(dstA, s.Option)
	switch n & 3 {
	case 1: return hash.StringHash(uint32(n), s.SrcA[(n >> 2) % len(s.SrcA)])
	case 2: return hash.StringHash(uint32(n), dstA[(len(dstA) - 1 + (n >> 2)) % len(dstA)])
	case 3: return hash.StringHash(uint32(n), srcCut[(len(srcCut) - 1 + (n >> 2)) % len(srcCut)])
	case 0:	return hash.StringHash(uint32(n), srcFut[(len(srcFut) - 1 + (n >> 2)) % len(srcFut)])
	}
/*
	n %= len(s.SrcA) + len(s.DstA) + len(s.SrcCut) + len(s.SrcFut)
	if n < len(s.SrcA) {
		return hash.StringHash(1, s.SrcA[n])
	}
	n -= len(s.SrcA)
	if n < len(s.DstA) {
		return hash.StringHash(2, s.DstA[n])
	}
	n -= len(s.DstA)
	if n < len(s.SrcCut) {
		return hash.StringHash(3, s.SrcCut[n])
	}
	n -= len(s.SrcCut)
	if n < len(s.SrcFut) {
		return hash.StringHash(4, s.SrcFut[n])
	}
*/
	return 0
}

func (s *NewSample) Parity() uint16 {
	return 0
}
func (s *NewSample) Output() uint16 {
	if s.Out {
		return 1
	}
	return 0
}

func (n *NewSample) Key() [3]string {
	return [3]string{strings.Join(n.DstA, "\x00"), strings.Join(n.SrcA, "\x00"), n.Option}
}

func (n *NewSample) OldSample() Sample {
	i := n.I
	j := n.J
	dstA := n.DstA
	for len(dstA) < n.Len {
		dstA = append(dstA, "")
	}
	srcA := n.SrcA
	for len(srcA) < n.Len {
		srcA = append(srcA, "")
	}
	return Sample{
		hash.StringsHash(0, srcA[1*i/2:i+j/2]),
		hash.StringsHash(0, srcA[2*i/3:i+j/3]),
		hash.StringsHash(0, srcA[4*i/5:i+j/5]),
		hash.StringsHash(0, srcA[6*i/7:i+j/11]),
		hash.StringsHash(0, srcA[10*i/11:i+j/11]),
		hash.StringsHash(0, dstA[0:i]),
		hash.StringsHash(0, srcA),
		hash.StringsHash(0, dstA[0:4*i/7]),
		hash.StringsHash(0, dstA[4*i/7:6*i/7]),
		hash.StringsHash(0, dstA[6*i/7:i]),
		hash.StringsHash(0, srcA[i:i+j/7]),
		hash.StringsHash(0, srcA[i+j/7:i+3*j/7]),
		hash.StringsHash(0, srcA[i+3*j/7:i+j]),
		hash.StringHash(0, n.Option),
	}
}

type Sample [14]uint32

func (s *Sample) Feature(n int) uint32 {
	a := hash.Hash(uint32(n), 0, 13)
	/*
		b := n % 28
		if b >= a {
			b++
		}
	*/
	return s[a] /*+ s[b]*/ + s[13]
}
func (s *Sample) Parity() uint16 {
	return 0
}
func (s *Sample) Output() uint16 {
	return 0
}


type Output bool

func (s *Output) Feature(n int) uint32 {
	if *s {
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
		if len(columns) != 2 {
			fmt.Println("Line does not have exactly two columns:", line)
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

func NewDataset(filename string) (out map[[3]string]*NewSample) {
	out = make(map[[3]string]*NewSample)
	var oneway = make(map[string]string)
	var multiway = make(map[string]map[string]struct{})
	loop(filename, func(src, dst string) {
		srca := strings.Split(src, " ")
		dsta := strings.Split(dst, " ")
		if len(srca) != len(dsta) {
			return
		}
		for i := range srca {
			if multiway[srca[i]] == nil {
				multiway[srca[i]] = make(map[string]struct{})
			}
			multiway[srca[i]][dsta[i]] = struct{}{}
		}
	})

	//data, _ := json.Marshal(multiway)
	//fmt.Println(string(data))

	for k, mmap := range multiway {
		for v := range mmap {
			if len(mmap) == 1 {
				oneway[k] = v
				delete(multiway, k)
				break
			}
		}
	}
	loop(filename, func(src, dst string) {
		srca := strings.Split(src, " ")
		dsta := strings.Split(dst, " ")
		if len(srca) != len(dsta) {
			return
		}
		for i, srcv := range srca {
			if _, ok := oneway[srcv]; ok {
				continue
			}
			for option := range multiway[srcv] {
			
				if option != dsta[i] {
					j := len(srca) - i
					s := &NewSample{
						SrcA: srca,
						DstA: dsta[0:i],
						SrcCut: srca[0:i],
						SrcFut: srca[i:],
						Option: option,
						Out: false,
						I: i,
						J: j,
						Len: len(srca),
					}
					out[s.Key()] = s
				}
			}
		}
	})
	loop(filename, func(src, dst string) {
		srca := strings.Split(src, " ")
		dsta := strings.Split(dst, " ")
		if len(srca) != len(dsta) {
			return
		}
		for i, srcv := range srca {
			if _, ok := oneway[srcv]; ok {
				continue
			}
			j := len(srca) - i
			option := dsta[i]
			s := &NewSample{
				SrcA: srca,
				DstA: dsta[0:i],
				SrcCut: srca[0:i],
				SrcFut: srca[i:],
				Option: option,
				Out: true,
				I: i,
				J: j,
				Len: len(srca),
			}
			out[s.Key()] = s
		}
	})
	return
}

func Split(out map[[3]string]*NewSample) (data []NewSample) {
	for _, v := range out {
		data = append(data, *v)
	}
	return
}
