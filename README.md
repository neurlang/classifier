# Neurlang Binary Classifier

* A machine learning algorithm for training binary classifiers of integers.
  * The integers should be uint32. If your samples are larger, pre-hash them (for instance using murmur).
* It works best when the size of the true set and the false set is almost the same.
  * If they aren't, random numbers are padded to the smaller set 
* The resulting models are very fast (they are hash based)

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

If your solution is found, you can slowly decrease the Subtractor to get smaller solutions.

### Printer

If you are annoyed by Backtracking printed too often, then increase the Printer

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

Works starting with golang 1.13

## License

Apache 2.0 or Public Domain at your option
