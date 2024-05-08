package mnist

import "os"
import "fmt"
import "crypto/sha256"
import "io"
import "compress/gzip"
import "bytes"
import "strings"
import "math/rand"

func userHomeDir() string {
	dirname, err := os.UserHomeDir()
	if err != nil {
		return "~"
	}
	return dirname + "/"
}

const tmpDirectory = `/tmp/mnist/`
var customDirectory = userHomeDir() + `/go/src/example.com/repo.git/classifier/datasets/mnist/`
var searchDirectories = []string{tmpDirectory, customDirectory}

const inferSetImg = "t10k-images-idx3-ubyte.gz"
const inferSetVal = "t10k-labels-idx1-ubyte.gz"
const trainSetImg = "train-images-idx3-ubyte.gz"
const trainSetVal = "train-labels-idx1-ubyte.gz"
const inferDigImg = "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
const inferDigVal = "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
const trainDigImg = "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
const trainDigVal = "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"

// original
const ImgSize = 28
var TrainSet, InferSet [][ImgSize*ImgSize]byte
var TrainLabels, InferLabels []byte

// downscaled
const SmallImgSize = 13
var SmallTrainSet, SmallInferSet [][SmallImgSize*SmallImgSize]byte


type SmallInput [SmallImgSize*SmallImgSize]byte
func (i *SmallInput) Feature(n int) uint32 {
	n %= ((SmallImgSize-1)*(SmallImgSize-1))
	return uint32(i[n]) | uint32(i[n+1])<<8 | uint32(i[n+SmallImgSize])<<16 | uint32(i[n+1+SmallImgSize])<<24
}



type Input [ImgSize*ImgSize]byte
func (i *Input) Feature(n int) uint32 {
	n %= ((ImgSize-1)*(ImgSize-1))
	return uint32(i[n]) | uint32(i[n+1])<<8 | uint32(i[n+ImgSize])<<16 | uint32(i[n+1+ImgSize])<<24
}


// Shuffle shuffles the mnist train dataset
func ShuffleTrain() {
	rand.Shuffle(len(TrainLabels), func(i, j int) {
		TrainLabels[i], TrainLabels[j] = TrainLabels[j], TrainLabels[i]
		TrainSet[i], TrainSet[j] = TrainSet[j], TrainSet[i]
		SmallTrainSet[i], SmallTrainSet[j] = SmallTrainSet[j], SmallTrainSet[i]
	})
}
// Shuffle shuffles the mnist infer dataset
func ShuffleInfer() {
	rand.Shuffle(len(InferLabels), func(i, j int) {
		InferLabels[i], InferLabels[j] = InferLabels[j], InferLabels[i]
		InferSet[i], InferSet[j] = InferSet[j], InferSet[i]
		SmallInferSet[i], SmallInferSet[j] = SmallInferSet[j], SmallInferSet[i]
	})
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

var success byte
var globalErr error
func Error() error {
	// are 4 files loaded?
	if success == 4 {
		return nil
	}
	if globalErr == nil {
		return fmt.Errorf("Unknown mnist dataset error/bug")
	}
	return globalErr
}

func init() {
	var files = map[string]string{
		inferDigImg: inferSetImg,
		inferDigVal: inferSetVal,
		trainDigImg: trainSetImg,
		trainDigVal: trainSetVal,
	}
outer:
	for _, dir := range searchDirectories {
		for hash, name := range files {
			if _, err := os.Stat(dir+name); err == nil {
				f, err := os.Open(dir+name)
				if err != nil {
					globalErr = fmt.Errorf("Cannot open file to check file '%s': %e", dir+name, err)
					continue outer
				}
				h := sha256.New()
				_, err = io.Copy(h, f)
				f.Close()
				if err != nil {
					globalErr = fmt.Errorf("Cannot copy file to hash file '%s': %e", dir+name, err)
					continue outer
				}
				if fmt.Sprintf("%x", h.Sum(nil)) != hash {
					globalErr = fmt.Errorf("File hash for file '%s' is incorrect", dir+name)
				}
				f, err = os.Open(dir+name)
				if err != nil {
					globalErr = fmt.Errorf("Cannot open file to ungzip file '%s': err", dir+name, err)
					continue outer
				}
				gzipReader, err := gzip.NewReader(f)
				if err != nil {
					f.Close()
					globalErr = fmt.Errorf("Gzip file '%s' Error: %e", dir+name, err)
					continue outer
				}
				var uncompressedBuffer bytes.Buffer
				_, err = uncompressedBuffer.ReadFrom(gzipReader)
				gzipReader.Close()
				f.Close()
				if err != nil {
					globalErr = fmt.Errorf("Buffering file '%s' Error: %e", dir+name, err)
					continue outer
				}
				
				// Get the uncompressed data as a byte slice
				uncompressedData := uncompressedBuffer.Bytes()

				var isTrainFile = strings.HasPrefix(name, "train-")
				var isImgFile = strings.Contains(name, "-images-")

				if isImgFile {
					// skip header
					uncompressedData = uncompressedData[16:]

					var numberImages = len(uncompressedData) / (ImgSize*ImgSize)

					var set = make([][ImgSize*ImgSize]byte, numberImages, numberImages)
					var smallSet = make([][SmallImgSize*SmallImgSize]byte, numberImages, numberImages)

					for i := range set {
						var ptr = (ImgSize*ImgSize)*i
						copy(set[i][:], uncompressedData[ptr:])

						var small [SmallImgSize*SmallImgSize]byte
						for y := 0; y < SmallImgSize; y++ {
							for x := 0; x < SmallImgSize; x++ {
								var base = ptr+1+ImgSize
								small[y*SmallImgSize+x] = max4(
									uncompressedData[base+(2*x)+(2*y*ImgSize)], 
									uncompressedData[base+(2*x)+(2*y*ImgSize)+1],
									uncompressedData[base+(2*x)+(2*y*ImgSize)+ImgSize], 
									uncompressedData[base+(2*x)+(2*y*ImgSize)+ImgSize+1],
								)
							}
						}
						copy(smallSet[i][:], small[:])
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
					
					var set = uncompressedData

					if isTrainFile {
						TrainLabels = set
					} else {
						InferLabels = set
					}
				}

				success++

			} else if os.IsNotExist(err) {
				globalErr = fmt.Errorf("File '%s' does not exist", dir+name)
				continue outer
			} else {
				globalErr = fmt.Errorf("Error checking if file '%s' exists: %e", dir+name, err)
				continue outer
			}
		}
	}

}
