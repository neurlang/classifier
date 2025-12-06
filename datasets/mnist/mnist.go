// Package MNIST is the 60000 + 10000 handwritten digits dataset
package mnist

import "os"
import "fmt"
import "crypto/sha256"
import "io"
import "compress/gzip"
import "bytes"
import "strings"
import "github.com/neurlang/classifier/datasets/stringhash"

func userHomeDir() string {
	dirname, err := os.UserHomeDir()
	if err != nil {
		return "~"
	}
	return dirname + "/"
}

// original
const ImgSize = 28

// downscaled
const SmallImgSize = 13

// SmallInput is the type of small input image
type SmallInput struct {
	Image [SmallImgSize * SmallImgSize]byte
	Label byte
}

// Feature extracts the n-th feature from small input
func (i *SmallInput) Feature(n int) uint32 {
	return stringhash.ByteSample{Buf: i.Image[:]}.Feature(n)
}
func (i *SmallInput) Parity() uint16 {
	//return stringhash.BalancedByteSample{Buf: i.Image[:]}.Parity()
	return 0
}

func (i *SmallInput) Output() uint16 {
	return uint16(i.Label)
}

// Input is the type of input image
type Input struct {
	Image [ImgSize * ImgSize]byte
	Label byte
}

// Feature extracts the n-th feature from input
func (i *Input) Feature(n int) uint32 {
	return stringhash.ByteSample{Buf: i.Image[:]}.Feature(n)
}
func (i *Input) Parity() uint16 {
	//return stringhash.BalancedByteSample{Buf: i.Image[:]}.Parity()
	return 0
}

func (i *Input) Output() uint16 {
	return uint16(i.Label)
}

func max4(a, b, c, d byte) (o byte) {
	o = a
	if b > o {
		o = b
	}
	if c > o {
		o = c
	}
	if d > o {
		o = d
	}
	return o
}

func New() (TrainSet []Input, InferSet []Input, SmallTrainSet []SmallInput, SmallInferSet []SmallInput, err error) {
	var success byte
	var globalErr error

	const tmpDirectory = `/tmp/mnist/`

	var customDirectory = userHomeDir() + `/go/src/example.com/repo.git/classifier/datasets/mnist/`
	var rootDirectory = userHomeDir() + `/classifier/datasets/mnist/`
	var searchDirectories = []string{tmpDirectory, customDirectory, rootDirectory}

	const inferSetImg = "t10k-images-idx3-ubyte.gz"
	const inferSetVal = "t10k-labels-idx1-ubyte.gz"
	const trainSetImg = "train-images-idx3-ubyte.gz"
	const trainSetVal = "train-labels-idx1-ubyte.gz"
	const inferDigImg = "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
	const inferDigVal = "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
	const trainDigImg = "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
	const trainDigVal = "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"

	var files_hsh = map[string]string{
		inferSetImg: inferDigImg,
		inferSetVal: inferDigVal,
		trainSetImg: trainDigImg,
		trainSetVal: trainDigVal,
	}
	var files_ord = []string{
		inferSetImg,
		inferSetVal,
		trainSetImg,
		trainSetVal,
	}
outer:
	for _, dir := range searchDirectories {
		for _, name := range files_ord {
			hash := files_hsh[name]
			if _, err := os.Stat(dir + name); err == nil {
				f, err := os.Open(dir + name)
				if err != nil {
					globalErr = fmt.Errorf("Cannot open file to check file '%s': %v", dir+name, err)
					continue outer
				}
				h := sha256.New()
				_, err = io.Copy(h, f)
				f.Close()
				if err != nil {
					globalErr = fmt.Errorf("Cannot copy file to hash file '%s': %v", dir+name, err)
					continue outer
				}
				if fmt.Sprintf("%x", h.Sum(nil)) != hash {
					globalErr = fmt.Errorf("File hash for file '%s' is incorrect", dir+name)
				}
				f, err = os.Open(dir + name)
				if err != nil {
					globalErr = fmt.Errorf("Cannot open file to ungzip file '%s': %v", dir+name, err)
					continue outer
				}
				gzipReader, err := gzip.NewReader(f)
				if err != nil {
					f.Close()
					globalErr = fmt.Errorf("Gzip file '%s' Error: %v", dir+name, err)
					continue outer
				}
				var uncompressedBuffer bytes.Buffer
				_, err = uncompressedBuffer.ReadFrom(gzipReader)
				gzipReader.Close()
				f.Close()
				if err != nil {
					globalErr = fmt.Errorf("Buffering file '%s' Error: %v", dir+name, err)
					continue outer
				}

				// Get the uncompressed data as a byte slice
				uncompressedData := uncompressedBuffer.Bytes()

				var isTrainFile = strings.HasPrefix(name, "train-")
				var isImgFile = strings.Contains(name, "-images-")

				if isImgFile {
					// skip header
					uncompressedData = uncompressedData[16:]

					var numberImages = len(uncompressedData) / (ImgSize * ImgSize)

					var set = make([]Input, numberImages, numberImages)
					var smallSet = make([]SmallInput, numberImages, numberImages)

					for i := range set {
						var ptr = (ImgSize * ImgSize) * i
						copy(set[i].Image[:], uncompressedData[ptr:])

						var small [SmallImgSize * SmallImgSize]byte
						for y := 0; y < SmallImgSize; y++ {
							for x := 0; x < SmallImgSize; x++ {
								var base = ptr + 1 + ImgSize
								small[y*SmallImgSize+x] = max4(
									uncompressedData[base+(2*x)+(2*y*ImgSize)],
									uncompressedData[base+(2*x)+(2*y*ImgSize)+1],
									uncompressedData[base+(2*x)+(2*y*ImgSize)+ImgSize],
									uncompressedData[base+(2*x)+(2*y*ImgSize)+ImgSize+1],
								)
							}
						}
						copy(smallSet[i].Image[:], small[:])
					}

					if isTrainFile {
						TrainSet = set
						SmallTrainSet = smallSet
					} else {
						InferSet = set
						SmallInferSet = smallSet
					}
				} else {
					// skip header
					uncompressedData = uncompressedData[8:]

					for i, lbl := range uncompressedData {
						if isTrainFile {
							TrainSet[i].Label = lbl
							SmallTrainSet[i].Label = lbl
						} else {
							InferSet[i].Label = lbl
							SmallInferSet[i].Label = lbl
						}
					}

				}

				success++

			} else if os.IsNotExist(err) {
				globalErr = fmt.Errorf("File '%s' does not exist", dir+name)
				continue outer
			} else {
				globalErr = fmt.Errorf("Error checking if file '%s' exists: %v", dir+name, err)
				continue outer
			}
		}
	}

	if success != 4 {
		if globalErr == nil {
			return nil, nil, nil, nil, fmt.Errorf("Unknown mnist dataset error/bug")
		}
		return nil, nil, nil, nil, globalErr
	}

	return TrainSet, InferSet, SmallTrainSet, SmallInferSet, nil
}
