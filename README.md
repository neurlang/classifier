# Neurlang Binary Classifier

* A machine learning algorithm for training binary classifiers of integers.
  * The integers should be uint32. If your samples are larger, pre-hash them (for instance using murmur).
* It works best when the size of the true set and the false set is almost the same.
  * If they aren't, random numbers are padded to the smaller set 
* Maximum problem size is limited to 2 x 65536 values (2 sets of size 65536)
  * Big problems wont fit to ram likely
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

### InitialModulo

InitialModulo needs to be increased to the point when you get at least one row such as this:
`Size:  188 Modulo: 4097`
This means that the problem was reduced at least one time. Then, you can slowly decrease InitialModulo
to still be able to reduce the problem and to be able to find a smaller solution.

### Numerator, Denominator, Subtractor

If your problem is reduced a few times (`Size:  188 Modulo: 4097` was printed), but a solution is not found,
you need to increase the fraction.
For instance initially you used Numerator 20, Denominator 21, this means a fraction of 20/21 is in use.
You can increase this to, say 100/101. Subtractor you can keep at 1.

If your solution is found, you can slowly decrease the fraction to get smaller solutions.

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


## License

Apache 2.0 or Public Domain at your option
