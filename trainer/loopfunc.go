package trainer

import "os"
import "fmt"
import "math/rand"
import "time"

import "github.com/neurlang/classifier/net/feedforward"
import "github.com/neurlang/classifier/parallel"

func NewLoopFunc(net feedforward.FeedforwardNetwork, succ *int, treshold int, evaluate func() (int, [32]byte), trainWorst func([]int, int) (undo func())) func() {

	var m = parallel.NewMoveSet()
	var success, state = evaluate()
	if success >= 100 && treshold > 100 {
		println("Max accuracy or wrong data. Exiting")
		os.Exit(0)
	}
	var default_backoff = func() {
		println("Infinite loop - algorithm stuck in local minimum. Exiting")
		os.Exit(0)
	}
	backoff := default_backoff
	var local_minimums = make(map[[32]byte]struct{})
	fmt.Printf("%x\n", state)
	for {
		for infloop := 0; infloop < net.Len(); infloop++ {
			var shuf []int
			if success >= treshold {
				shuf = net.Sequence(false)
				rand.Seed(time.Now().UnixNano())
				rand.Shuffle(len(shuf), func(i, j int) { shuf[i], shuf[j] = shuf[j], shuf[i] })
			} else {
				shuf = net.Branch(false)
			}
			if m.Exists(state, shuf[0], byte(success)) {
				continue
			}
			for worst := 0; worst < len(shuf); worst++ {
				println("training #", worst, "hastron of", len(shuf), "hashtrons total")
				inSucc := success
				if succ != nil {
					inSucc = *succ
				}
				worsts := []int{shuf[worst]}
				if inSucc >= treshold {
					if worst+1 < len(shuf) {
						worsts = append(worsts, shuf[worst+1])
					} else {
						break
					}
				}
				if this_backoff := trainWorst(worsts, inSucc); this_backoff != nil {
					infloop = -1
					this_success, this_state := evaluate()
					if this_success >= 100 && treshold > 100 {
						println("Max accuracy or wrong data. Exiting")
						os.Exit(0)
					}
					if _, bad := local_minimums[this_state]; bad {
						this_backoff()
						break
					} else {
						backoff, success, state = this_backoff, this_success, this_state
					}
				} else if worst == 0 {
					break
				}
				fmt.Printf("%x\n", state)
				m.Insert(state, shuf[worst], byte(success))
				if worst != len(shuf)-1 {
					if m.Exists(state, shuf[worst+1], byte(success)) {
						break
					}
				}
			}
		}
		local_minimums[state] = struct{}{}
		backoff()
		backoff = default_backoff
		success, state = evaluate()
		if success >= 100 && treshold > 100 {
			println("Max accuracy or wrong data. Exiting")
			os.Exit(0)
		}
	}
}
