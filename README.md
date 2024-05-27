# Neurlang Binary Classifier (Hashtron)

* A machine learning algorithm for training binary classifiers of integers.
  * The integers should be uint32. If your samples are larger, pre-hash them (for instance using murmur).
  * Doesn't use floating point at all
* It works best when the size of the true set and the false set is almost the same.
  * If they aren't, random numbers are padded to the smaller set 
* The resulting models are very fast (they are hash based)
* Larger hashtron networks can also be trained
  * A MNIST handwritten digits demo is currently provided
  * The larger models don't need to be fully connected, reducing the need for large scale matrix multiplication.

# NEW: AI Antivirus demo and dataset

It was figured out that this model is successful for virus classification based on TLSH hashes.
The success rate we archieved was 94% on the training set. We are currently trying to improve this result further.

## Usage - Training

1. Clone the repo
2. Compile, Run:
```
cd cmd/train_is_alnum/
go build
./train_is_alnum
```
3. Open solutions.txt and copy your solution into the cmd/infer_is_alnum/program.go

## Usage - Inference

1. Compile, Run:
```
cd cmd/infer_is_alnum/
go build
./infer_is_alnum
```
2. You can see that for each character it was calculated whether it is alpha numeric.

## Adjusting for a custom dataset

You can adjust the hyper parameters for a custom dataset

### Factor

Factor is initially set to value such as 1. Later, when you get correct solution, you can experimentally increase it to higher values to get a smaller solution to your problem.

### Subtractor

Subtractor is initially set to value such as 1.

If your solution is found, you can slowly increase the Subtractor to get smaller solutions.

### Printer

If you are annoyed by Backtracking printed too often, then increase the Printer

## Multi classifier

You can also train a N-way (instead of binary) classifier. Simply use the Datamap.
Datamap is `map[uint16][uint64]` internally. You can fill it with custom data and then
initialize the hyperparameters and train as usual.

```
	h.Training(datamap)
```

Multi classifier also needs the programBits variable:

```

type Program [][2]uint32

func (p Program) Len() int {
	return len(p)
}
func (p Program) Get(n int) (uint32, uint32) {
	return (p)[n][0], (p)[n][1]
}
func (p Program) Bits() byte {
	return programBits
}
```

Then init and use the model:

```
	var model = Program(program)

	var predicted = inference.Uint16Infer(input, model)
```

One important thing to keep in mind, if your multi way Datamap contains few equivalence
groups and the values are sparse, you can get much better performing model if you reduce
your datamap.

### Reducing the datamap

You will be actually training two models, the first is trained as:
```
	h.Training(datamap.Reduce(true))
```
This is the first model and takes longer to train. Then you train the second model:

```
	h.Training(datamap.Reduce(false))
```
Finally, during inference, supply the bits for each model:

```
func (p Program) Bits() byte {
	if &p[0] == &program0[0] {
		return program0Bits
	}
	return program1Bits
}
```
And you can do inference:

```
	var predicted = inference.Uint16Infer2(input, model1, model0)
```


## Related algorithms

* Locality Sensitive Hashing (LSH): LSH is a technique used for solving the approximate or exact Near Neighbor Search problem in high-dimensional spaces. It's often employed in binary classification tasks, where it can be used to efficiently find similar items or data points.
* Logistic Regression: Despite its name, logistic regression is a linear model for binary classification. It models the probability that a given sample belongs to a particular class using the logistic function.
* Support Vector Machines (SVM): SVM is a powerful algorithm for binary classification that finds the hyperplane that best separates the classes while maximizing the margin between them. It can handle both linear and non-linear classification tasks through the use of different kernel functions.
* Decision Trees: Decision trees recursively split the feature space into regions, with each region corresponding to a specific class label. These splits are chosen to maximize the information gain or minimize impurity at each step.
* Random Forests: A random forest is an ensemble method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
* Gradient Boosting Machines (GBM): GBM builds an ensemble of weak learners (typically decision trees) sequentially, with each new learner focusing on the mistakes of the previous ones. It combines their predictions to produce a strong predictive model.
* K-Nearest Neighbors (KNN): KNN is a non-parametric method that classifies a sample based on the majority class among its k nearest neighbors in the feature space. It's simple but can be computationally expensive, especially with large datasets.
* Neural Networks: Neural networks, particularly deep learning models, have gained popularity for binary classification tasks due to their ability to automatically learn complex patterns from data. Common architectures include feedforward neural networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).
* Naive Bayes: Naive Bayes classifiers are based on Bayes' theorem and assume that the features are conditionally independent given the class label. Despite their simplicity and the "naive" assumption, they often perform well in practice, especially with text classification tasks.
* Adaptive Boosting (AdaBoost): AdaBoost is another ensemble method that combines multiple weak classifiers (often decision trees) to create a strong classifier. It iteratively adjusts the weights of misclassified samples to focus on the difficult cases.

## Compatibility

Works starting with golang 1.13 / golang 1.16 + cu dependency for CUDA learning

## License

Apache 2.0 or Public Domain at your option
