# Neurlang Binary Classifier (Hashtron)

![Neurlang Binary Classifier](./classifier.svg)

Neurlang Classifier is a machine learning algorithm designed for training binary classifiers of integers. It is particularly efficient for tasks where the input data consists of `uint32` integers. If your samples are larger than `uint32`, pre-hashing (e.g., using murmur hash) is recommended.

## Key Features

- **Integer-based Classification**: The algorithm works directly with `uint32` integers, avoiding the use of floating-point arithmetic.
- **Balanced Datasets**: Performs best when the true and false sets are balanced. If they are not, you can use `Parity()` to balance the data.
- **Hash-based Models**: The resulting models are hash-based, making them extremely fast for inference.
- **Scalable Networks**: Supports larger networks, including those with majority pooling layers, reducing the need for large-scale matrix multiplication.

## Implementing a Dataset

To implement a dataset for training the network, define a slice of samples where each sample has the following methods:

- `Feature(int) uint32`: Returns the feature at the specified index. Each front layer hashtron reads its index starting from 0.
- `Parity() uint16`: Returns the parity of the sample. This is used to balance the dataset. If your classes contain equal number of samples, you can return 0.
- `Output() uint16`: Returns the output label. This is the prediction for the sample.

## Implementing a Network

Here is an example of how to implement a network with majority pooling layers:

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

- `fanout1` and `fanout3` define the majority pooling dimensions.
- `fanout2` and `fanout4` define the number of hashtrons.
- The final layer contains one hashtron for predictions.
  - The `0` in the final layer can be replaced by the number of bits the network should predict (up to 16 supported).
  - `0` or `1` means 1 bit is predicted.

### Training and Inference

- Use `net.Tally4(sample, ...)` to tally samples during training.
- Use `net.Infer2(sample)` to predict values from the network.

## Compatibility

Neurlang Classifier is compatible with Go versions 1.13 and above. For CUDA-based learning, Go 1.16 with CUDA dependencies is required.

## License

Neurlang Classifier is licensed under Apache 2.0 or Public Domain, at your option.
