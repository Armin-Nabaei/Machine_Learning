Here is Python code to implement interpolation search:

```python
def interpolation_search(sorted_list, target):
    low = 0
    high = len(sorted_list) - 1
    
    while low <= high:
        mid = low + int(((float(high - low) / (sorted_list[high] - sorted_list[low])) * (target - sorted_list[low])))
        
        if sorted_list[mid] == target:
            return mid
        
        if target < sorted_list[mid]:
            high = mid - 1
        else:
            low = mid + 1
            
    return -1
```

The key steps are:

1. Initialize low and high pointers to the bounds of the sorted list

2. Calculate the probe position `mid` using interpolation between low and high

3. Compare the target to the element at mid, and update low or high pointers accordingly

4. Repeat steps 2-3 until target is found or low > high (target not found)

5. Return index of target if found, else -1

This implements the interpolation search algorithm, which uses a probe position calculated from an interpolation between the low and high bounds. The runtime is O(log log n) in the average case.