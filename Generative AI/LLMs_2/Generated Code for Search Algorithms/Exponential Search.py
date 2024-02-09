Here is Python code to implement Exponential Search:

```python
import math

def exponential_search(arr, x):
  
  # Find range for binary search by 
  # repeated doubling
  if arr[0] == x:
    return 0
  
  i = 1
  while i < len(arr) and arr[i] <= x:
    i = i * 2

  # Do binary search in found range    
  return binary_search(arr, i/2, min(i, len(arr)-1), x)

def binary_search(arr, l, r, x):

  if r >= l:

    mid = l + (r - l) // 2

    # If found at mid, return mid
    if arr[mid] == x:
      return mid
    
    # Search left half if x is smaller   
    elif arr[mid] > x:
      return binary_search(arr, l, mid-1, x)

    # Search right half  
    else:
      return binary_search(arr, mid + 1, r, x)
  
  else:
    return -1
```

This implements exponential search by first finding the range to do binary search on by repeated doubling of the index. Once the range is found, regular binary search is done in that range to find the element x.