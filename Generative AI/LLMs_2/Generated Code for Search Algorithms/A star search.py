Here is Python code to implement A* search:

```python
import heapq

class Node:
    def __init__(self, state, parent, cost, heuristic):
        self.state = state
        self.parent = parent 
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def astar(start, goal):
    frontier = []
    heapq.heappush(frontier, Node(start, None, 0, heuristic(start, goal)))
    explored = set()

    while frontier:
        current = heapq.heappop(frontier)
        
        if current.state == goal:
            path = []
            while current.parent:
                path.append(current.state)
                current = current.parent
            path.reverse()
            return path
        
        if current.state not in explored:
            explored.add(current.state)
            for neighbor in neighbors(current.state):
                cost = current.cost + 1 # Assume 1 cost per step
                heuristic = heuristic(neighbor, goal)
                node = Node(neighbor, current, cost, heuristic)
                heapq.heappush(frontier, node)

    return None # No path found

def heuristic(a, b):
    # Use your heuristic function here
    return 0 

def neighbors(state):
    # Return list of neighbors for a given state
    return []

start = (0, 0) 
goal = (5, 5)
path = astar(start, goal)
print(path)
```

This implements the A* search algorithm by using a priority queue (heapq) to store the frontier of nodes to explore. It calculates the f-cost as the cost to reach the node plus a heuristic estimate of the cost to the goal. The heuristic function and neighbors function would need to be defined for a specific problem.