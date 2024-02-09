Here is Python code to implement jump search:

```python
import math

def jump_search(arr, x):
  n = len(arr)
  
  # Finding block size to be jumped
  step = math.sqrt(n)
  
  # Finding the block where element is present (if it is present)
  prev = 0
  while arr[int(min(step, n)-1)] < x:
    prev = step
    step += math.sqrt(n)
    if prev >= n:
      return -1
  
  # Doing a linear search for x in block beginning with prev.
  while arr[int(prev)] < x:
    prev += 1
    
    # If we reached next block or end of array, element is not present.
    if prev == min(step, n):
      return -1
  # If element is found
  if arr[int(prev)] == x:
    return prev
  
  return -1
```

This implements jump search by first calculating the block size to jump based on the square root of the array length. It then jumps through blocks, searching linearly within a block once the potential block is found. It returns the index if element is found, else -1.