# Neurlang Classifier

![Neurlang Binary Classifier](./classifier.svg)

Neurlang Classifier is a lightweight ML library for binary and quaternary neural networks that train quickly on CPUs. It models neurons with simple integer-to-boolean filters, enabling networks to be trained purely with integer arithmetic—no backpropagation, no floating-point math, and no GPU required. This makes training fast on multi-core CPUs while keeping dependencies minimal.

The framework has been proven in production, including training large-scale transformers for the [goruut phonemizer](https://github.com/neurlang/goruut). Use cases include virus detection, handwritten digit recognition, phoneme modeling, speech command classification, and more.

## Features

- **No backpropagation or perceptrons**: Uses simple integer-based logic instead of weight gradients, enabling an alternative ML paradigm
- **CPU-optimized, hardware-light**: Requires no GPU—training is fast on multi-core CPUs using bitwise and integer operations
- **Quaternary neurons**: Implements custom layers (convolution, attention, pooling, parity, etc.) that operate on boolean/integer data
- **Tiny dependencies**: Written in pure Go with minimal external libraries, simplifying installation and portability
- **Hash-based models**: Resulting models are extremely fast for inference using hash-based feature extraction
- **Proven at scale**: Already used in production projects to train large-scale transformers

## Getting Started

### Prerequisites

- Go 1.18 or higher

### Installation

```bash
go get github.com/neurlang/classifier
```

## Usage Examples

### Training MNIST Digit Classifier

```bash
cd cmd/train_mnist
go run .
```

### Running Inference on MNIST

```bash
cd cmd/infer_mnist
go run .
```

### Training Virus Detection Classifier

```bash
cd cmd/train_is_virus
go run .
```

### Other Examples

The `cmd/` directory contains additional demo programs:
- `train_is_alnum` / `infer_is_alnum` - Alphanumeric character classification
- `train_speak` - Speech command recognition
- `train_squareroot` / `infer_squareroot` - Mathematical function learning
- `train_phonemizer_multi` / `train_phonemizer_ulevel` - Grapheme-to-phoneme conversion

Run `./cmd/trainall.sh` to train all examples or `./cmd/runall.sh` to run all inference demos.

## Package Overview

- **cmd** - Demo programs with `train_*` and `infer_*` commands for various tasks
- **datasets** - Core dataset interface and implementations:
  - `isalnum` - Alphanumeric character dataset
  - `isvirus` - TLSH file hash signatures for virus detection
  - `mnist` - Standard MNIST handwritten digits (60k train / 10k test)
  - `phonemizer_multi` / `phonemizer_ulevel` - Grapheme-to-phoneme datasets
  - `speak` - Speech commands dataset
  - `squareroot` - Synthetic dataset for numeric relations
  - `stringhash` - String hashing and classification
- **hash** - Fast modular hash function implementation used by Neurlang layers
- **hashtron** - Core "hashtron" classifier model implementing the neuron logic
- **layer** - Abstract interfaces and implementations:
  - `conv2d` - 2D binary convolutional layer
  - `crossattention` - Cross-attention layer for transformer-like models
  - `full` - Fully connected (dense) layer
  - `majpool2d` - 2D majority pooling layer
  - `parity` - Parity (XOR-like) layer
  - `sochastic` - Stochastic/randomly connected layer
  - `sum` - Element-wise sum layer
- **net** - Network architecture definitions:
  - `feedforward` - Feedforward network architecture
- **parallel** - Concurrency utilities (`ForEach`, `LoopUntil`) to speed up training
- **trainer** - High-level training orchestration managing training loops over datasets

## Implementing a Dataset

To implement a dataset, define a slice of samples where each sample has these methods:

```go
type Sample interface {
    Feature(int) uint32  // Returns the feature at the specified index
    Parity() uint16      // Returns parity for dataset balancing (0 if balanced)
    Output() uint16      // Returns the output label/prediction
}
```

## Implementing a Network

Example network with majority pooling layers:

```go
const fanout1 = 3
const fanout2 = 5
const fanout3 = 3
const fanout4 = 5

var net feedforward.FeedforwardNetwork
net.NewLayerP(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
net.NewCombiner(majpool2d.MustNew(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
net.NewLayerP(fanout1*fanout2, 0, 1<<fanout2)
net.NewCombiner(majpool2d.MustNew(fanout2, 1, fanout1, 1, fanout2, 1, 1))
net.NewLayer(1, 0)
```

- `fanout1` and `fanout3` define majority pooling dimensions
- `fanout2` and `fanout4` define the number of hashtrons
- The final layer contains one hashtron for predictions (0 or 1 means 1 bit predicted, up to 16 bits supported)

### Training and Inference

Training uses the `trainer` package with custom evaluation and training functions:

```go
import "github.com/neurlang/classifier/trainer"
import "github.com/neurlang/classifier/parallel"

// Define training function
trainWorst := trainer.NewTrainWorstFunc(net, nil, nil, nil,
    func(worst []int, tally datasets.AnyTally) {
        parallel.ForEach(len(dataslice), 1000, func(i int) {
            var sample = dataslice[i]
            net.AnyTally(&sample, worst, tally, customErrorFunc)
        })
    })

// Define evaluation function
evaluate := trainer.NewEvaluateFunc(net, len(dataslice), 99, &improved_success_rate, dstmodel,
    func(length int, h trainer.EvaluateFuncHasher) int {
        // Evaluate accuracy on dataset
        return successRate
    })

// Run training loop
trainer.NewLoopFunc(net, &improved_success_rate, 100, evaluate, trainWorst)()
```

Inference is straightforward:

```go
predicted := net.Infer2(&sample)  // Returns predicted output
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

Neurlang Classifier is licensed under Apache 2.0 or Public Domain, at your option.
