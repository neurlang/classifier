// Package isvirus implements the IsVirus TLSH file Hashes machine learning Dataset
package isvirus

import "fmt"
import "encoding/hex"
import "crypto/rand"

type Input [35]byte

func (i *Input) Feature(n int) uint32 {
	m1 := n % 35
	m2 := n % 34
	if m2 >= m1 {
		m2++
	}
	return uint32(i[m1]) | uint32(i[m2])<<8
}

func (i *Input) Parity() bool {
	parity := false

	for _, b := range *i {
		for b > 0 {
			if b&1 == 1 {
				// Toggle the parity bool
				parity = !parity
			}
			b >>= 1
		}
	}

	return parity
}

type Output bool

func (i *Output) Feature(n int) uint32 {
	if *i {
		return 1
	}
	return 0
}

var Inputs []Input
var Outputs []Output

func randomize() (value [35]byte) {
	rand.Read(value[:])
	return
}
func decode(hexString string) (value [35]byte) {
	// Decode hex string to bytes
	bytes, err := hex.DecodeString(hexString)
	if err != nil {
		fmt.Println("Error decoding hex string:", err)
		return
	}

	copy(value[:], bytes)
	return value
}

func Balance() {
	// Loop through true random (clean)
	for i := len(clean); i < len(virus); i++ {
		Inputs = append(Inputs, randomize())
		Outputs = append(Outputs, false)
	}
}

func init() {
	// Loop through virus
	for _, v := range virus {

		//if i == 1000 {break;}

		Inputs = append(Inputs, decode(v))
		Outputs = append(Outputs, true)

	}
	// Loop through clean
	for _, v := range clean {

		//if i == 1000 {break;}

		Inputs = append(Inputs, decode(v))
		Outputs = append(Outputs, false)

	}
}
