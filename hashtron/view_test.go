package hashtron

import "testing"
import "bytes"
import "fmt"

func FuzzHashtronSerialize(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, buffer []byte) {
		var dualBuffer [][2]uint32
		for i, v := range buffer {
			if i%8 == 0 {
				dualBuffer = append(dualBuffer, [2]uint32{0, 0})
			} else if i%4 == 0 {
				dualBuffer[len(dualBuffer)-1][1] = dualBuffer[len(dualBuffer)-1][0]
				dualBuffer[len(dualBuffer)-1][0] = 0
			}
			dualBuffer[len(dualBuffer)-1][0] |= uint32(v)
			dualBuffer[len(dualBuffer)-1][0] <<= 8
		}
		if len(dualBuffer) == 0 {
			return
		}
		var tron, err = New(dualBuffer, 0)
		if err != nil {
			panic(err.Error())
		}
		if len(tron.program) != len(dualBuffer) {
			panic("len mismatch:" + fmt.Sprint(len(tron.program)) + "!=" + fmt.Sprint(len(dualBuffer)) + ":" + fmt.Sprint(dualBuffer))
		}
		for i, v := range tron.program {
			if dualBuffer[i][0] != v[0] {
				panic(i)
			}
			if dualBuffer[i][1] != v[1] {
				panic(i)
			}
		}
		var buf bytes.Buffer
		err = tron.WriteJson(&buf)
		if err != nil {
			panic(err.Error())
		}
		str := buf.String()
		tron.program = nil
		buf.WriteString(str)
		err = tron.ReadJson(&buf)
		if err != nil {
			panic(err.Error())
		}
		if len(tron.program) != len(dualBuffer) {
			panic("len mismatch:" + fmt.Sprint(len(tron.program)) + "!=" + fmt.Sprint(len(dualBuffer)) + ":" + str)
		}
		for i, v := range tron.program {
			if dualBuffer[i][0] != v[0] {
				panic(str + ":" + fmt.Sprint(i) + "[0]:" + fmt.Sprint(v[0]) + "!=" + fmt.Sprint(dualBuffer[i][0]))
			}
			if dualBuffer[i][1] != v[1] {
				panic(str + ":" + fmt.Sprint(i) + "[1]:" + fmt.Sprint(v[1]) + "!=" + fmt.Sprint(dualBuffer[i][1]))
			}
		}
	})
}
