package hashtron

import "errors"
import "bytes"
import "io"

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

// WriteJson serializes hashtron into a pre-xorred json io.Writer
func (h Hashtron) WriteJson(b io.Writer, eol ...byte) error {
	var open = []byte("[")
	var clos = []byte("]")
	var comm = []byte(",")
	var minu = []byte("-")
	var quot = []byte(`"`)
	_, err := b.Write(open)
	if err != nil {
		return err
	}
	_, err = b.Write(eol)
	if err != nil {
		return err
	}
	var xor uint32
	var sub uint32
	for i, v := range h.program {
		if i != 0 {
			_, err = b.Write(comm)
		}
		if err != nil {
			return err
		}
		_, err = b.Write(open)
		if err != nil {
			return err
		}
		_, err = b.Write(intToBuf(v[0] ^ xor))
		if err != nil {
			return err
		}
		_, err = b.Write(comm)
		if err != nil {
			return err
		}
		if sub == 0 {
			_, err = b.Write(intToBuf(v[1]))
			sub = v[1]
		} else {
			if sub < v[1] {
				_, err = b.Write(minu)
				if err != nil {
					return err
				}
				_, err = b.Write(intToBuf(v[1] - sub))
			} else {
				_, err = b.Write(intToBuf(sub - v[1]))
			}
		}
		if err != nil {
			return err
		}
		_, err = b.Write(clos)
		if err != nil {
			return err
		}
		_, err = b.Write(eol)
		if err != nil {
			return err
		}
		xor = v[0]
		sub = v[1]
	}
	if len(h.quaternary) > 0 {
		if len(h.program) > 0 {
			_, err = b.Write(comm)
			if err != nil {
				return err
			}
		}
		_, err = b.Write(quot)
		if err != nil {
			return err
		}
		for _, by := range h.quaternary {
			bytes := [2]byte{by >> 4, by & 15}
			for _, d := range bytes {
				if d < 10 {
					_, err = b.Write([]byte{'0' + d})
					if err != nil {
						return err
					}
				} else {
					_, err = b.Write([]byte{'a' + d - 10})
					if err != nil {
						return err
					}
				}
			}
		}
		_, err = b.Write(quot)
		if err != nil {
			return err
		}
	}
	_, err = b.Write(clos)
	if err != nil {
		return err
	}
	_, err = b.Write(eol)
	if err != nil {
		return err
	}
	return nil
}

// ReadJson deserializes hashtron from a pre-xorred json io.Reader
func (h *Hashtron) ReadJson(b io.Reader) error {
	var number, number0, number1, xor, add uint32
	var buf [1]byte
	var inside, neg, quot, qlast, isnum bool
	h.program = nil
	for {
		_, err := b.Read(buf[0:1])
		if err != nil {
			return err
		}
		switch buf[0] {
		case '"':
			inside = false
			qlast = false
			quot = !quot
			number = 0
			isnum = false
		case '-':
			neg = true
		case '[':
			inside = true
			number = 0
		case 'a', 'b', 'c', 'd', 'e', 'f':
			number *= 16
			number += uint32(buf[0] - 'a' + 10)
			if qlast {
				h.quaternary = append(h.quaternary, byte(number))
				number = 0
			}
			qlast = !qlast
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			isnum = true
			if quot {
				number *= 16
			} else {
				number *= 10
			}
			number += uint32(buf[0] - '0')
			if quot {
				if qlast {
					h.quaternary = append(h.quaternary, byte(number))
					number = 0
				}
				qlast = !qlast
			}
		case ']':
			if !inside {
				if add == 0 {
					add = number1
				} else {
					if neg {
						add += number1
					} else {
						add -= number1
					}
				}
				if isnum {
					h.program = append(h.program, [2]uint32{number0 ^ xor, add})
				}
				return nil
			}
			inside = false
			number1 = number
			number = 0
		case ',':
			if !inside {
				if add == 0 {
					add = number1
				} else {
					if neg {
						add += number1
					} else {
						add -= number1
					}
				}
				if isnum {
					h.program = append(h.program, [2]uint32{number0 ^ xor, add})
				}
				xor ^= number0
			} else {
				number0 = number
			}
			number = 0
			neg = false
		}
	}
}
