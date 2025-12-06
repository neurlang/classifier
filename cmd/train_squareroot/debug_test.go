package main

import (
"fmt"
"github.com/neurlang/classifier/datasets/squareroot"
"github.com/neurlang/classifier/net/feedforward"
)

func main() {
dataset := squareroot.Medium()

var net feedforward.FeedforwardNetwork
net.NewLayer(squareroot.MediumClasses, 10)

fmt.Printf("Network has %d hashtrons\n", net.Len())
fmt.Printf("GetLastCells: %d\n", net.GetLastCells())
fmt.Printf("GetBits: %d\n", net.GetBits())
fmt.Printf("GetClasses: %d\n", net.GetClasses())

// Test a few samples
for i := 0; i < 5; i++ {
io := squareroot.Sample(dataset[i])
predicted := net.Infer2(&io)
expected := io.Output()
fmt.Printf("Input: %4d, Expected: %2d (0b%05b), Predicted: %2d (0b%05b)\n", 
i, expected, expected, predicted, predicted)
}
}
