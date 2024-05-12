package hashtron

import "errors"
import "bytes"

// intToBuf converts integer into a buffer
func intToBuf(n uint32) (buf []byte) {
	var buffer [10]byte
	buf = buffer[:]
	for i := range buf {
		buf[9-i] = byte(n%10) + '0'
		n /= 10
	}
	for len(buf) > 1 && buf[0] == '0' {
		buf = buf[1:]
	}
	return
}

// isNameChar reports whether the name of hashtron is valid
func isNameChar(c rune) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || (c == '_')
}

// BytesBuffer serializes hashtron into a golang code program
func (h Hashtron) BytesBuffer(name string, eol ...byte) (b *bytes.Buffer, err error) {
	if len(eol) == 0 || len(eol) == 1 || len(eol) == 2 {
		if len(eol) != 0 && !(eol[0] == '\r' || eol[0] == '\n' || eol[0] == ';') {
			return nil, errors.New("eol 1 is invalid")
		}
		if len(eol) == 2 && !(eol[1] == '\n' || eol[1] == ' ') {
			return nil, errors.New("eol 2 is invalid")
		}
	} else {
		return nil, errors.New("eol is invalid")
	}
	for _, v := range name {
		if !isNameChar(v) {
			return nil, errors.New("name is invalid")
		}
	}
	b = new(bytes.Buffer)
	b.WriteString("var program")
	b.WriteString(name)
	b.WriteString("Bits byte = ")
	b.Write(intToBuf(uint32(h.bits)))
	b.Write(eol)
	b.WriteString("var program")
	b.WriteString(name)
	b.WriteString(" = [][2]uint32{")
	if len(eol) == 2 && eol[0] == ';' {
		b.WriteByte(eol[1])
	} else {
		b.Write(eol)
	}
	for _, v := range h.program {
		if !(len(eol) == 2 && eol[0] == ';') {
			b.WriteByte('\t')
		}
		b.WriteByte('{')
		b.Write(intToBuf(v[0]))
		b.WriteByte(',')
		b.Write(intToBuf(v[1]))
		b.WriteString("},")
		if len(eol) == 2 && eol[0] == ';' {
			b.WriteByte(eol[1])
		} else {
			b.Write(eol)
		}
	}
	b.WriteByte('}')
	if len(eol) == 2 && eol[1] == ' ' {
		b.WriteByte(eol[0])
	} else {
		b.Write(eol)
	}
	return
}
