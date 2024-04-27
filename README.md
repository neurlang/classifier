# Neurlang Binary Classifier

* A machine learning algorithm for training binary classifiers of integers.
  * The integers should be uint32. If your samples are larger, pre-hash them (for instance using murmur).
* It works best when the size of the true set and the false set is almost the same.
  * If they aren't, random numbers are padded to the smaller set 
* Maximum problem size is limited to 2 x 65536 values (2 sets of size 65536)
  * Big problems wont fit to ram likely
* The resulting models are very fast (they are hash based)
