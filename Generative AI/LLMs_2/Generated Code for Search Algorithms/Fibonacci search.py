Here is Python code to implement Fibonacci search:

```python
import math

def fibonacci_search(arr, x):
  n = len(arr)

  # Initialize fibonacci numbers
  fib2 = 0 # (m-2)'th Fibonacci No.
  fib1 = 1 # (m-1)'th Fibonacci No. 
  fibM = fib2 + fib1 # m'th Fibonacci

  # fibM is going to store the smallest Fibonacci 
  # Number greater than or equal to n
  while (fibM < n):
    fib2 = fib1
    fib1 = fibM
    fibM = fib2 + fib1

  # Marks the eliminated range from front
  offset = -1;

  # while there are elements to be inspected.
  while (fibM > 1):
    # Check if fibMm2 is valid location
    i = min(offset+fib2, n-1)

    # If x is greater than the value at index fibMm2
    if (arr[i] < x):
      fibM = fib1
      fib1 = fib2
      fib2 = fibM - fib1
      offset = i

    # If x is less than the value at index fibMm2
    elif (arr[i] > x):
      fibM = fib2
      fib1 = fib1 - fib2
      fib2 = fibM - fib1

    # element found. return index
    else: 
      return i

  # comparing the last element with x  
  if(fib1 and arr[offset+1] == x):
    return offset+1;

  #element not found. return -1
  return -1
```

This implements Fibonacci search on the given array arr to search for element x. It uses Fibonacci numbers to efficiently narrow down the search space at each iteration. The key steps are:

1. Initialize Fibonacci numbers fib2, fib1, fibM.

2. Find smallest Fibonacci number greater than or equal to array length n. 

3. Mark eliminated range by offset.

4. At each iteration, compare x with arr[offset+fibMm2] to eliminate elements on one side.

5. Decrement Fibonacci numbers and update offset.

6. Repeat until fibM becomes 1. 

7. Return index if element found, else -1.

So this allows searching a sorted array in O(log n) time complexity using Fibonacci numbers.