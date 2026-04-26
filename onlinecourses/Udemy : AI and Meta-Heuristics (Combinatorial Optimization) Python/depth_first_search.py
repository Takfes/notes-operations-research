"Graph traversal algortihms - Depth First Search"


class Node:
    def __init__(self, name):
        self.name = name
        self.adjacency_list = []
        self.visited = False


def depth_first_search(start_node):
    # LIFO structure
    stack = [start_node]
    # Mark the start node as visited
    start_node.visited = True

    # Keep looping until the stack is empty
    while stack:
        # Get last element from the stack
        current_node = stack.pop()
        # Set the current node as visited
        current_node.visited = True
        print(current_node.name)

        # Consider neighbors of the current node
        for n in current_node.adjacency_list:
            if not n.visited:
                n.visited = True
                stack.append(n)


if __name__ == "__main__":
    # Create nodes
    node1 = Node("A")
    node2 = Node("B")
    node3 = Node("C")
    node4 = Node("D")
    node5 = Node("E")

    # # Define the neighbors of each node
    # node1.adjacency_list.append(node2)
    # node1.adjacency_list.append(node3)
    # node2.adjacency_list.append(node4)
    # node4.adjacency_list.append(node5)

    # Define the neighbors of each node
    node1.adjacency_list.append(node2)
    node1.adjacency_list.append(node5)
    node2.adjacency_list.append(node3)
    node2.adjacency_list.append(node4)
    node3.adjacency_list.append(node4)

    # Perform the breadth-first search
    depth_first_search(node1)
