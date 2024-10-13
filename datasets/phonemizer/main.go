package phonemizer

import "github.com/neurlang/classifier/hash"
import (
	"bufio"
	"fmt"
	"os"
	"strings"
	//"encoding/json"
)

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

func NewDataset(filename string) (out map[Sample]bool) {
	out = make(map[Sample]bool)
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
					out[[...]uint32{
						hash.StringsHash(0, srca[1*i/2:i+j/2]),
						hash.StringsHash(0, srca[2*i/3:i+j/3]),
						hash.StringsHash(0, srca[4*i/5:i+j/5]),
						hash.StringsHash(0, srca[6*i/7:i+j/11]),
						hash.StringsHash(0, srca[10*i/11:i+j/11]),
						hash.StringsHash(0, dsta[0:i]),
						hash.StringsHash(0, srca),
						hash.StringsHash(0, dsta[0:4*i/7]),
						hash.StringsHash(0, dsta[4*i/7:6*i/7]),
						hash.StringsHash(0, dsta[6*i/7:i]),
						hash.StringsHash(0, srca[i:i+j/7]),
						hash.StringsHash(0, srca[i+j/7:i+3*j/7]),
						hash.StringsHash(0, srca[i+3*j/7:i+j]),
						hash.StringHash(0, option),
					}] = false
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
			out[[...]uint32{
				hash.StringsHash(0, srca[1*i/2:i+j/2]),
				hash.StringsHash(0, srca[2*i/3:i+j/3]),
				hash.StringsHash(0, srca[4*i/5:i+j/5]),
				hash.StringsHash(0, srca[6*i/7:i+j/11]),
				hash.StringsHash(0, srca[10*i/11:i+j/11]),
				hash.StringsHash(0, dsta[0:i]),
				hash.StringsHash(0, srca),
				hash.StringsHash(0, dsta[0:4*i/7]),
				hash.StringsHash(0, dsta[4*i/7:6*i/7]),
				hash.StringsHash(0, dsta[6*i/7:i]),
				hash.StringsHash(0, srca[i:i+j/7]),
				hash.StringsHash(0, srca[i+j/7:i+3*j/7]),
				hash.StringsHash(0, srca[i+3*j/7:i+j]),
				hash.StringHash(0, option),
			}] = true
		}
	})
	return
}

func Split(out map[Sample]bool) (keys []Sample, values []bool) {
	for k, v := range out {
		keys = append(keys, k)
		values = append(values, v)
	}
	return
}
