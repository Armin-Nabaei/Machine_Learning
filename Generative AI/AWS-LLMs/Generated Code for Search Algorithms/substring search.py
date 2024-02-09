Here is Python code to implement substring search:

```python
def substring_search(text, pattern):
  """
  Check if pattern exists in text as a substring.
  Return the index of the start of the first occurrence if found, 
  or -1 if not found.
  """
  
  n = len(pattern)
  for i in range(len(text) - n + 1):
    if text[i:i+n] == pattern:
      return i
  return -1

text = "hello world" 
pattern = "world"
print(substring_search(text, pattern)) # 6

text = "hello python"
pattern = "java" 
print(substring_search(text, pattern)) # -1
```

This code loops through the text and checks if the substring of length n starting at index i matches the pattern. It returns the first index where the pattern is found, or -1 if not found.