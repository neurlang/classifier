# Binary Classifier Learning Algorithm Proof of Convergence

## Dense Case

```
Binomial(2^n,2^(n-1)) * 0.5 * (2 ^ n) * 0.5 = (2^n * (2^n - 1)/2) * x
```

Where:
- \( x \) is the odd Catalan number (A038003).

**Left Side Explanation:**
- How many possible n-bit Boolean functions have half of their bits set to true (pseudo-random Boolean functions).
- Multiplied by 0.5 (it doesn't matter if a n-bit Boolean function swaps true and false).
- Multiplied by the number of possible salts to attempt to reduce the problem.
- Multiplied by 0.5 (each salt has a 50% chance of reducing the problem).

**Right Side Explanation:**
- Without the \( x \), how many possible n-bit learning problems exist when reduced to the result.
- \( x \) represents the odd Catalan number, which indicates how many solutions exist.

If this holds true, it would mean that every learning problem is solvable by the given learning algorithm, meaning it can learn any Boolean function with half of its output bits set to true.

In other words, the number of possible functions the algorithm can learn = the number of problems * the number of solutions.

| n | Left Side | Right Side | 4 * Odd Catalan Number | Odd Catalan Number (A038003) |
|---|-----------|------------|------------------------|-----------------------------|
| 1 | 2         | 1          | 4                      | 1                           |
| 2 | 6         | 6          | 4                      | 1                           |
| 3 | 70        | 28         | 20                     | 5                           |
| 4 | 12870     | 120        | 1716                   | 429                         |
| 5 | 601080390 | 496        | 38779380               | 9694845                     |
| 6 | 1832624140942590534 | 2016 | 58178544156907636     | 14544636039226909          |
| 7 | (unknown) | (unknown)  | (unknown)              | 94295850558771979787935384946380125 |

### For \( n = 1 \)

0 → 1, 1 → 0  
1 → 0, 0 → 1

2 functions.

**Learning Task**: Learn to distinguish between 0 and 1, regardless of order.

2 salts (salt 0 or 1, factor: \( 2^n \)).

1 1-bit learning problem.

1 odd Catalan number (everything has a solution).

```
2 * 0.5 * 2 * 0.5 = 1 * 1
```

### For \( n = 2 \)

6 functions for 2-bit number to 1 bit.

\[
0 → 0, 1 → 0, 2 → 1, 3 → 1 \
0 → 0, 1 → 1, 2 → 0, 3 → 1 \
0 → 1, 1 → 0, 2 → 0, 3 → 1 \
0 → 0, 1 → 1, 2 → 1, 3 → 0 \
0 → 1, 1 → 0, 2 → 1, 3 → 0 \
0 → 1, 1 → 1, 2 → 0, 3 → 0
\]

Number of salts \( 2^4 = |{0, 1, 2, 3}| = 4 \).

**Learning Task**: Learn to distinguish, regardless of order on each side:

\[
(0,1|2,3) \
(0,2|1,3) \
(0,3|1,2) \
(2,3|0,1) \
(1,3|0,2) \
(1,2|0,3)
\]

6 tasks.

1 odd Catalan number (everything has a solution).

```
6 * 0.5 * 4 * 0.5 = 6 * 1
```

### For \( n = 3 \)

```
70 * 0.5 * 8 * 0.5 = 28 * 5
140 = 28 * 5
```

## Sparse Case:

Let \( m \) (where `(m <= 2^(n-1))`) be the number of true inputs for the Boolean function, and \( 2m \) be the total number of inputs.  
This can be solved in polynomial time in \( m \), specifically `O(m^Constant)`

## Unbalanced Case:

There is a transformation rule for problems where the outputs of the desired algorithm are not balanced.  
Assuming the inputs are pseudo-random (if not, they can be hashed more).  
Take one bit from the input and XOR it with the output bit; this should balance the function more.
Apply learning algorithm yielding a model.
Then, on inference, un XOR the same input bit from the inference output and return the unbalanced output.
