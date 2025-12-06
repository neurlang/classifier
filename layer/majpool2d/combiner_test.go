package majpool2d

import rand "math/rand/v2"

import (
	"testing"
)

// every passes test
func TestEvery(t *testing.T) {
	const fanout1 = 1
	const fanout2 = 13
	const fanout3 = 1
	const fanout4 = 13

	a := MustNew2(1, fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 0)

outer:
	for {

		var combiner = a.Lay()
		var q = rand.IntN(fanout1 * fanout2 * fanout4)

		combiner.Put(q, true)

		println("put at", q)

		for j := 0; j < fanout2; j++ {
			if combiner.Feature(j) != 0 {
				println(q)
				break outer
			}
		}
		panic("repeatinga")
	}
	b := MustNew2(1, fanout2, 1, fanout1, 1, fanout2, 1, 0)
outer2:
	for {

		var combiner = b.Lay()
		var q = rand.IntN(fanout2)

		combiner.Put(q, true)

		println("put at", q)

		for j := 0; j < 1; j++ {
			if combiner.Feature(j) != 0 {
				println(q)
				break outer2
			}
		}
		panic("repeatingb")
	}

	c := MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0)

outer3:
	for {

		var combiner = c.Lay()
		var q = rand.IntN(fanout1 * fanout2 * fanout4)

		combiner.Put(q, true)

		println("put at", q)

		for j := 0; j < fanout2; j++ {
			if combiner.Feature(j) != 0 {
				println(q)
				break outer3
			}
		}
		panic("repeatingc")
	}
	d := MustNew2(fanout2, 1, fanout1, 1, fanout2, 1, 1, 0)
outer4:
	for {

		var combiner = d.Lay()
		var q = rand.IntN(fanout2)

		combiner.Put(q, true)

		println("put at", q)

		for j := 0; j < 1; j++ {
			if combiner.Feature(j) != 0 {
				println(q)
				break outer4
			}
		}
		panic("repeatingd")
	}
}

// every passes test
func TestBigPooling(t *testing.T) {
	const fanout1 = 5
	const fanout2 = 4
	const fanout3 = 3
	const fanout4 = 2
	a := MustNew2(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1, 0)
	var combiner = a.Lay()

	// Define the neuron index to test (neuron [1])
	neuronIndex := 1

	// Calculate the positions of the three majpooled inputs for neuron [1]
	// Assuming the inputs are grouped in sets of fanout3 (3 inputs per neuron)
	inputBase := neuronIndex * fanout2
	inputPositions := []int{inputBase, inputBase + 1, inputBase + 2}

	// Set all three majpooled inputs to true
	for _, pos := range inputPositions {
		combiner.Put(pos, true)
	}

	// Verify that the output of neuron [1] is true
	output := combiner.Feature(neuronIndex)
	if output == 0 {
		t.Errorf("Neuron [%d] output is false, expected true", neuronIndex)
	} else {
		println(output)
		t.Logf("Neuron [%d] output is %d, as expected", neuronIndex, output)
	}
}
