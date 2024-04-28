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

## License

Apache 2.0 or Public Domain at your option
