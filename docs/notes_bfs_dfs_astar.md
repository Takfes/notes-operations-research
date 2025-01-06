### Breadth-First Search (BFS)
- Uses a queue (FIFO) data structure.
- Explores all neighbors at the present depth before moving on to nodes at the next depth level.
- Time Complexity: O(V + E) where V is vertices and E is edges.
- Space Complexity: O(V) for the queue.
- Guarantees the shortest path in an unweighted graph.

### Depth-First Search (DFS)
- Uses a stack (LIFO) data structure.
- Explores as far as possible along each branch before backtracking.
- Can be implemented using recursion.
- Time Complexity: O(V + E) where V is vertices and E is edges.
- Space Complexity: O(V) for the stack.

### A* Search Algorithm
- Finds the shortest path from a start node to a goal node.
- Combines features of both DFS and BFS.
- It does this by breaking down into two steps : **f(n) = g(n) + h(n)**
- **f(n)**: Total estimated cost of the cheapest solution; from start to the final goal.
- **g(n)**: Represents the known cost to reach the current node; from the start node to node n.
- **h(n)**: Heuristic estimate of the cost from node n to the goal.
- Uses a priority queue to explore nodes.
- Time Complexity: O(E) where E is the number of edges.
- Space Complexity: O(V) where V is the number of vertices.