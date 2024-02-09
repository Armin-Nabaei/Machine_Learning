Here is Python code to implement greedy search:

```python
import heapq

def greedy_search(start, successors, is_goal, heuristic):
    frontier = [(heuristic(start), start)]
    explored = set()
    
    while frontier:
        _, state = heapq.heappop(frontier)
        if is_goal(state):
            return state
        explored.add(state)
        
        for successor in successors(state):
            if successor not in explored:
                heapq.heappush(frontier, (heuristic(successor), successor))
    
    return None
```

This implements a generic greedy search algorithm in Python. It takes in:

- `start` - the initial state
- `successors` - a function that generates valid successor states for a given state
- `is_goal` - a function that checks if a given state is a goal state
- `heuristic` - a heuristic function that estimates the cost to reach the goal from a given state

It uses a priority queue `frontier` to track the next most promising node based on the heuristic. At each step it pops the node with the lowest heuristic value, checks if it is the goal, and if not expands it to generate successor states to add to the frontier. It returns the goal state if found, else None.

The `explored` set keeps track of already expanded nodes to avoid duplicates. This implements a simple greedy best-first search guided by the heuristic function.