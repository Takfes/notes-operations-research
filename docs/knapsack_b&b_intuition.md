
# Knapsack 0/1 Branch And Bound process summary and key components

<br>
<br>
<br>


## my summary
- items = sorted list of items based on their density value
- nodes = encapsulates/represents the state, what has happened so far in this branch (items included/decisions made, accumulated value and weight so far) and an optimistic evaluation `bound` for the future considering the so far excluded and remaining (not yet decided for) items.
- queue = where we keep Nodes for future exploration. queue prioritizes most promising nodes first. we initialize this with the root node
- we start processing nodes from the queue, in a repetitive manner, until there are no more nodes in the queue. queue is responsible to give its best candidate to evaluate every time.
- how to process a node : 
    1. check whether this node stands a chance to improve the result - if its optimistic evaluation is already lower than the current value, then skip to the next item from the queue.
    2. recall nodes level. level is a pointer to the `sorted` items list. we store the level of the node as we put them in the queue. so an item that we process now, we had stored it's level by the point we had put them in the queue.
    3. increment the level by one. this now points to the next item from the `sorted` list. for that next item, generate two child scenarios out of the current node : 1) node state + include this item 2) node state + exclude this item. before putting them in the queue, update all their values accordingly.
- you can imagine that within every `while loop` iteration, we start from a nodes and we expand by one step. we recall where that node was left off wrt the `sorted` list, pick the next item, generate two future states (incl and excl) and put in the priority queue for future processing. 
- every iteration, we may jump around the `traversal tree` prioritizing the most promising node/state - that is priority's queue (max heap) responsibility, i.e. to prioritize promising states. that way, we rush to establish a better bound which in turn facilitates better future pruning

<br>
<br>
<br>

## generated summary - refined based on the above

### Refined Summary
1. **Items**:
   - A **sorted list** of items based on their **value-to-weight ratio** (density). This sorting ensures that when performing the linear relaxation (bound calculation), the greedy approach works optimally.

2. **Nodes**:
   - Each **node** encapsulates a **state** in the decision tree. It represents:
     - Decisions made so far (e.g., which items are included/excluded).
     - The **current value** and **weight** of the knapsack.
     - The **level**: an index pointing to the current position in the sorted items list.
     - An **upper bound (bound)**: an optimistic estimate of the maximum value achievable from this state onward.
   - Nodes guide exploration by representing "what-if" scenarios.

3. **Priority Queue**:
   - A data structure (implemented as a max-heap via `heapq`) used to manage **nodes waiting for exploration**.
   - Ensures that nodes with the **highest bound** (most promising potential) are processed first.
   - Starts with a **root node** representing no decisions made yet (empty knapsack).

4. **Main Process**:
   - The algorithm processes nodes from the priority queue **iteratively** until no more nodes remain.
   - Each iteration evaluates the most promising node, potentially expanding it into two child nodes based on whether the next item is included or excluded.

### **Processing a Node**
1. **Pruning (Does This Node Stand a Chance?)**:
   - Check if the node's **bound** (optimistic future value) is less than or equal to the **current best value** (best known solution so far). 
   - If so, skip this node (prune the branch).

2. **Recall Node’s Level**:
   - The **level** tells us the position of this node in the sorted list of items.
   - This ensures that nodes follow the same sequence as the sorted items.

3. **Generate Child Nodes**:
   - **Increment the level** by 1 to consider the next item in the sorted list.
   - For the next item:
     - **Include the item**: Create a new node reflecting the state after adding this item.
     - **Exclude the item**: Create a new node reflecting the state without this item.
   - **Update all values** (bound, value, weight) for the new nodes before adding them to the queue.

4. **Expand the Tree**:
   - Each iteration effectively "expands" the tree one level deeper for the most promising branch.
   - Newly generated child nodes are added to the queue, which prioritizes exploration based on their **bound**.

### **Traversal and Exploration**
- The **priority queue (max-heap)** ensures that the algorithm explores **high-potential branches first**:
  - This strategy "rushes" to establish a strong **current best value**, allowing more aggressive pruning of suboptimal branches.
  - Nodes with lower bounds are delayed or discarded entirely, saving computational effort.

- While the algorithm may appear to "jump around" in the traversal tree, this is intentional:
  - It doesn’t follow a depth-first or breadth-first approach but rather a **best-bound-first** approach. This dynamic prioritization accelerates finding better solutions.