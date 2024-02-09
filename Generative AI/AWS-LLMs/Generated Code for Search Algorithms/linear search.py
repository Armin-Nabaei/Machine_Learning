Here is Python code to implement linear search:

```python
def linear_search(list, target):
    """
    Returns the index position of the target if found, else returns None
    """

    for i in range(0, len(list)):
        if list[i] == target:
            return i
    return None

def verify(index):
    if index is not None:
        print("Target found at index: ", index)
    else:
        print("Target not found in list")

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

result = linear_search(numbers, 12)
verify(result)

result = linear_search(numbers, 6) 
verify(result)
```

This implements a linear search algorithm that iterates through the given list and checks each element to see if it matches the target. If found, it returns the index position. If not found, it returns None. 

The verify function prints out a message indicating if the target was found or not, based on whether the index returned was None or an actual index number.

To use it, simply call linear_search() passing the list to search and the target to find, store the result, then pass it to the verify() function to print the output.