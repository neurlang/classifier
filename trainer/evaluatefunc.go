package trainer

import "fmt"
import "math"

import "github.com/neurlang/classifier/parallel"
import "github.com/neurlang/classifier/net/feedforward"

type dummy struct{}

func (d dummy) MustPutUint16(n int, value uint16) {}
func (d dummy) Sum() [32]byte {
	return [32]byte{}
}

type EvaluateFuncHasher interface {
	MustPutUint16(n int, value uint16)
	Sum() [32]byte
}

// sampleSize calculates the statistically sufficient sample size
// for a given dataset size N and significance level (0–100).
func sampleSize(N int, significance byte) int {

	// Convert significance level to Z-score
	z := zScoreFromAlpha(100 - significance)

	// Assume worst-case proportion p = 0.5 for max variability
	p := 0.5
	e := float64(100 - significance) * 0.01 // Margin of error = 5%

	numerator := math.Pow(z, 2) * p * (1 - p)
	denominator := math.Pow(e, 2)

	// Initial sample size without population correction
	ss := numerator / denominator

	// Apply finite population correction
	correctedSS := ss * float64(N) / (float64(N) - 1 + ss)

	if int(correctedSS) > N {
		return N
	}

	return int(correctedSS)
}

// zScoreFromAlpha returns the Z-score for a given alpha level
// Common: 90% => 1.645, 95% => 1.96, 99% => 2.576
func zScoreFromAlpha(alpha byte) float64 {
	switch {
	case alpha <= 1:
		return 2.576 // 99% confidence
	case alpha <= 5:
		return 1.96 // 95% confidence
	case alpha <= 10:
		return 1.645 // 90% confidence
	default:
		return 1.96 // default fallback
	}
}

func NewEvaluateFunc(net feedforward.FeedforwardNetwork, length int, significance byte, succ *int, dstmodel *string,
	testFunc func(portion int, h EvaluateFuncHasher) int) func() (int, [32]byte) {

	return func() (int, [32]byte) {
		var h dummy
		var ha EvaluateFuncHasher = h
		var success int
		if length != 0 {
			var l = length
			if (succ != nil && (*succ < 99 && *succ > 0)) || (succ == nil) {
				l = sampleSize(length, significance)
			}
			hsh := parallel.NewUint16Hasher(l)
			ha = hsh
			success = testFunc(l, hsh)
		} else {
			success = testFunc(0, h)
		}

		if dstmodel == nil || *dstmodel == "" {
			err := net.WriteZlibWeightsToFile("output." + fmt.Sprint(success) + ".json.t.lzw")
			if err != nil {
				println(err.Error())
			}
		}

		if dstmodel != nil && len(*dstmodel) > 0 && ((succ != nil && (*succ < success || success >= 99)) || succ == nil) {
			if succ != nil && *succ > 0 {
				err := net.WriteZlibWeightsToFile(*dstmodel)
				if err != nil {
					println(err.Error())
				}
			}
			if succ != nil {
				*succ = success
			}
		} else if dstmodel != nil && len(*dstmodel) > 0 {
			if succ != nil {
				*succ = success
			}
		}

		return success, ha.Sum()
	}
}
