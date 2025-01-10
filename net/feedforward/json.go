package feedforward

import "compress/lzw"
import "compress/zlib"
import "io"
import "os"

// WriteCompressedWeightsToFile writes model weights to a lzw file
func (f *FeedforwardNetwork) WriteCompressedWeightsToFile(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	err = f.WriteCompressedWeights(file)
	file.Close()
	return err
}

// WriteCompressedWeights writes model weights to a writer
func (f *FeedforwardNetwork) WriteCompressedWeights(w io.Writer) error {
	lw := lzw.NewWriter(w, lzw.LSB, 8)

	_, err := lw.Write([]byte("[\n"))
	if err != nil {
		return err
	}
	for i := 0; i < f.Len(); i++ {
		if i != 0 {
			_, err = lw.Write([]byte(",\n"))
			if err != nil {
				return err
			}

		}
		err := f.GetHashtron(i).WriteJson(lw)
		if err != nil {
			return err
		}
	}
	_, err = lw.Write([]byte("]\n"))
	if err != nil {
		return err
	}
	return lw.Close()
}

// ReadCompressedWeightsFromFile reads model weights from a lzw file
func (f *FeedforwardNetwork) ReadCompressedWeightsFromFile(name string) error {
	file, err := os.Open(name)
	if err != nil {
		return err
	}
	err = f.ReadCompressedWeights(file)
	file.Close()
	return err
}

// ReadCompressedWeights reads model weights from a reader
func (f *FeedforwardNetwork) ReadCompressedWeights(w io.Reader) error {
	lw := lzw.NewReader(w, lzw.LSB, 8)

	for i := 0; i < f.Len(); i++ {

		var buf [1]byte

		for buf[0] != '[' {
			_, err := lw.Read(buf[0:1])
			if err != nil {
				return err
			}
		}

		err := f.GetHashtron(i).ReadJson(lw)
		if err != nil {
			return err
		}
	}
	return lw.Close()
}

// WriteZlibWeightsToFile writes model weights to a lzw file
func (f *FeedforwardNetwork) WriteZlibWeightsToFile(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	err = f.WriteZlibWeights(file)
	file.Close()
	return err
}

// WriteZlibWeights writes model weights to a writer
func (f *FeedforwardNetwork) WriteZlibWeights(w io.Writer) error {
	lw := zlib.NewWriter(w)

	_, err := lw.Write([]byte("[\n"))
	if err != nil {
		return err
	}
	for i := 0; i < f.Len(); i++ {
		if i != 0 {
			_, err = lw.Write([]byte(",\n"))
			if err != nil {
				return err
			}

		}
		err := f.GetHashtron(i).WriteJson(lw)
		if err != nil {
			return err
		}
	}
	_, err = lw.Write([]byte("]\n"))
	if err != nil {
		return err
	}
	return lw.Close()
}

// ReadZlibWeightsFromFile reads model weights from a lzw file
func (f *FeedforwardNetwork) ReadZlibWeightsFromFile(name string) error {
	file, err := os.Open(name)
	if err != nil {
		return err
	}
	err = f.ReadZlibWeights(file)
	file.Close()
	return err
}

// ReadZlibWeights reads model weights from a reader
func (f *FeedforwardNetwork) ReadZlibWeights(w io.Reader) error {
	lw, err := zlib.NewReader(w)
	if err != nil {
		return err
	}

	for i := 0; i < f.Len(); i++ {

		var buf [1]byte

		for buf[0] != '[' {
			_, err = lw.Read(buf[0:1])
			if err != nil {
				return err
			}
		}

		err := f.GetHashtron(i).ReadJson(lw)
		if err != nil {
			return err
		}
	}
	return lw.Close()
}
