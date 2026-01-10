package trainer

import "fmt"
import "runtime"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/quaternary"
import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/hashtron"

func NewTrainWorstFunc(net feedforward.FeedforwardNetwork, minpremodulo, premodulo, maxpremodulo *int,
	tallyFunc func(w []int, t datasets.AnyTally)) func(worst []int, succ int) (undo func()) {
	return func(worst []int, succ int) (undo func()) {

		if len(worst) == 0 {
			return nil
		}

		var tally = datasets.NewAnyTally(datasets.TallyType(len(worst)))
		if tally == nil {
			return nil
		}

		if premodulo != nil && *premodulo > 0 {
			tally.SetGlobalPremodulo(uint32(*premodulo))
		} else if minpremodulo != nil && *minpremodulo > 0 && maxpremodulo != nil && *maxpremodulo > 0 {
			const span = 50 * 50
			value := (100 - succ) * (100 - succ)
			premodulo := value*(*minpremodulo-*maxpremodulo)/span + *maxpremodulo
			if premodulo < 2 {
				premodulo = 2
			}
			tally.SetGlobalPremodulo(uint32(premodulo))
		}

		tallyFunc(worst, tally)

		if !tally.GetImprovementPossible() {
			return nil
		}

		var qq = make([][]byte, len(worst))

		fmt.Println("hashtron positions:", worst, "(job size:", tally.Len(), ")")

		// Calculate size check when len(worst) > 1
		if len(worst) > 1 {
			var previousSumLenQ int
			var newSumQ int

			for i, idx := range worst {
				ptr := net.GetHashtron(idx)
				previousSumLenQ += ptr.LenQ()

				dset := tally.DatasetAt(i)
				q := quaternary.Make(dset)
				newSumQ += len(q)

				qq[i] = q
			}

			// If new sum is bigger than previous sum, return nil
			if newSumQ > previousSumLenQ {
				return nil
			}
		} else {
			// Single hashtron case
			dset := tally.DatasetAt(0)
			q := quaternary.Make(dset)
			qq[0] = q
		}

		var restoreFns []func()

		for i, idx := range worst {
			ptr := net.GetHashtron(idx)
			q := qq[i]

			pmod := [][2]uint32{}
			if tally.IsGlobalPremodulo() {
				pmod = append(pmod, tally.GetGlobalSaltPremodulo())
			}

			htron, err := hashtron.New(pmod, ptr.Bits(), []byte(q))
			if err != nil {
				panic(err.Error())
			}
			backup := *ptr
			*ptr = *htron

			restoreFns = append(restoreFns, func() {
				*ptr = backup
			})
		}

		tally = nil
		runtime.GC()

		return func() {
			for _, fn := range restoreFns {
				fn()
			}
		}
	}
}
