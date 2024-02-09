Here is Python code to implement ubiquitous binary search:

```python
def ubiquitous_binary_search(arr, target):
  left = 0
  right = len(arr) - 1
  
  while left <= right:
    mid = (left + right) // 2
    
    if arr[mid] == target:
      return mid
    elif arr[mid] < target:
      left = mid + 1
    else:
      right = mid - 1
      
  return -1
```

This implements a binary search algorithm that can work on any sorted array `arr` to search for a `target` value. It follows the typical binary search approach:

- Initialize `left` and `right` pointers to the start and end of the array
- Calculate the `mid` index as the midpoint between `left` and `right` 
- Compare `arr[mid]` to the `target`. If equal, return the index.
- If not equal, update either `left` or `right` pointer based on if `target` is greater or less than element at `mid`.
- Repeat until `left > right`, at which point target is not found in array.
- Return -1 to indicate target not found.

The "ubiquitous" name refers to how this can work on any sorted array without needing to know specifics of the elements or sorting.