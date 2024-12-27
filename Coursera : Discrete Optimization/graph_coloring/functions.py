import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# Data Paths
ROOT_DIR = Path(__file__).resolve().parent
FOOTPRINTS = ROOT_DIR / "footprints.json"
SOLUTION_DIR = ROOT_DIR / "sols"
DATA_DIR = ROOT_DIR / "data"


"""
# ==============================================================
# file management functions
# ==============================================================
"""


# TODO : Adjust this to problem specific data
def parse_input_data(input_data, input_type="string"):
    """Parse a problem data from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.strip().split("\n")
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_nodes, n_edges = map(int, lines[0].split())

    # Parse problem data
    data_columns = ["e1", "e2"]
    lines = [x for x in lines if x != "\n"][1:]
    assert len(lines) == n_edges, f"Expected {n_edges} items, got {len(lines)}"
    dataset = pd.DataFrame(
        [list(map(int, x.split())) for x in lines],
        columns=data_columns,
        dtype=int,
    )

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "dataset": dataset,
    }


# TODO : Adjust this to problem specific data
def calculate_data_footprint(data):
    return (
        data["n_nodes"] + data["n_edges"] + data["dataset"].sum().sum().item()
    )


# TODO : Adjust this to problem specific data
def generate_output(obj_value, solution, optimized_indicator=0):
    obj = obj_value
    answer = f"{obj} {optimized_indicator}\n"
    answer += " ".join(map(str, solution))
    return answer


def generate_dummy_output():
    pass


def list_files_in_dir(dir_path=DATA_DIR, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return sorted(files, key=lambda s: (len(s), s))


def write_solution(contents, file_name, file_path=SOLUTION_DIR):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w") as file:
        file.write(contents)


def save_footprints_to_disk(dictionary, file_path=FOOTPRINTS):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def load_footprints_from_disk(file_path=FOOTPRINTS):
    with open(file_path) as file:
        return json.load(file)


def load_solution_from_disk(file_name, file_path=SOLUTION_DIR):
    with open(os.path.join(file_path, file_name)) as file:
        return file.read()


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper


"""
# ==============================================================
# solve_it template
# ==============================================================
"""

# from functions import fake_solver


def fake_solver(input_data):
    data = parse_input_data(input_data, input_type="string")
    footprints = load_footprints_from_disk()
    footprint = calculate_data_footprint(data)
    file_name = footprints[str(footprint)]
    output_data = load_solution_from_disk(file_name)
    return output_data


"""
# ==============================================================
# problem specific functions
# ==============================================================
"""

# TODO : https://www.coursera.org/learn/discrete-optimization/discussions/forums/R8Z6rVxEEea-8wq_-anwpw/threads/EKeqEnpMEe2RnwrIxQ9Aww


def plot_graph(edges, colors, node_size=250):
    G = nx.Graph()
    G.add_edges_from(edges)

    color_map = []
    for node in G:
        color_map.append(colors[node])

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        node_color=color_map,
        with_labels=True,
        node_size=node_size,
        font_color="white",
        edge_color="gray",
    )
    plt.show()


class Vertex:
    def __init__(self, index):
        self.index = index
        self.neighbors = []
        self.color = None

    def __repr__(self):
        return f"N({self.index}): c={self.color}, s={self.saturation}, d={self.degree}"

    def __str__(self):
        return f"N({self.index}): c={self.color}, s={self.saturation}, d={self.degree}"

    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)

    def set_color(self, color):
        self.color = color

    @property
    def neighbor_colors(self):
        return [neighbor.color for neighbor in self.neighbors]

    @property
    def saturation(self):
        return len({x for x in self.neighbor_colors if x is not None})

    @property
    def degree(self):
        return len(self.neighbors)


class DSatur:
    def __init__(self, n_nodes: int, edges: list[tuple[int, int]]):
        self.colors = []
        self.nodes = [Vertex(i) for i in range(n_nodes)]
        # self.nodes = [Vertex(i) for i in list(set(sum(edges, ())))]
        for e1, e2 in edges:
            self.nodes[e1].add_neighbor(self.nodes[e2])
            self.nodes[e2].add_neighbor(self.nodes[e1])

    @property
    def cost(self):
        return len(set(self.colors))

    def pick_a_color(self, node):
        available_colors = set(self.colors).difference(
            set(node.neighbor_colors)
        )
        if available_colors:
            return min(available_colors)
        else:
            self.colors.append(len(self.colors) + 1)
            return max(self.colors)

    @timeit
    def solve(self, verbose=False):
        q = [x for x in self.nodes]
        counter = 1
        while q:
            q.sort(key=lambda x: (x.saturation, x.degree), reverse=True)

            if verbose:
                print(
                    f"> Start iteration {counter}, len(q)={len(q)}, len(self.nodes)={len(self.nodes)}"
                )
                print("* Sorted list of nodes (q) --->")
                print(q)

            node = q.pop(0)
            color = self.pick_a_color(node)

            if verbose:
                print(f"* Selected node before setting color: {node}")

            node.set_color(color)

            if verbose:
                print(f"* Selected node after setting color: {node}")
                print("* Original -unsorted- list of nodes --->")
                print(self.nodes)
                print(
                    f"< End iteration {counter}, len(q)={len(q)}, len(self.nodes)={len(self.nodes)}"
                )
                print()

            counter += 1


"""
# ==============================================================
# Reference Implementation
# ==============================================================
"""


# class Color:
#     index: int
#     n_nodes: int

#     def __init__(self, index) -> None:
#         self.index = index
#         self.n_nodes = 0

#     def __repr__(self):
#         return f"C{self.index}"

#     def add_node(self):
#         self.n_nodes = self.n_nodes + 1


# class Node:
#     neighbors: list["Node"]
#     index: int
#     color: Color

#     def __init__(self, index):
#         self.index = index
#         self.neighbors = []
#         self.color = None

#     def __repr__(self) -> str:
#         return f"N{self.index}|{self.color}|{self.saturation}"

#     def add_neighbor(self, node: "Node"):
#         if node not in self.neighbors:
#             self.neighbors.append(node)

#     def set_color(self, color: Color):
#         self.color = color
#         color.add_node()

#     @property
#     def neighbor_colors(self):
#         return [n.color for n in self.neighbors if n.color is not None]

#     @property
#     def saturation(self):
#         return len(set(n.color for n in self.neighbors if n.color is not None))

#     @property
#     def degree(self):
#         return len(self.neighbors)


# class DSatur:
#     N: list[Node]
#     C: list[Color]
#     history: list[Node]

#     def __init__(self, nodes: list[int], edges: list[tuple[int, int]]):
#         N = [Node(i) for i in nodes]
#         for e in edges:
#             i, j = e
#             N[i].add_neighbor(N[j])
#             N[j].add_neighbor(N[i])
#         self.N = N
#         self.C = []
#         self.history = []

#     def find_next_color(self, node: Node) -> Color:
#         next_color = None
#         for c in self.C:
#             if c not in node.neighbor_colors:
#                 next_color = c
#                 break
#         if next_color is None:
#             next_color = Color(len(self.C) + 1)
#             self.C.append(next_color)
#         return next_color

#     @timeit
#     def solve(self, save_history=False):
#         Q = [n for n in self.N]  # Pool of uncolored nodes
#         counter = 0
#         while len(Q) > 0:
#             Q.sort(key=lambda x: (x.saturation, x.degree), reverse=True)
#             n: Node = Q.pop(0)
#             next_color = self.find_next_color(n)
#             n.set_color(next_color)
#             if save_history:
#                 self.history.append(n)
#             counter += 1
#         self.C.sort(key=lambda x: x.n_nodes, reverse=True)

#     @property
#     def cost(self):
#         return len(self.C)


# start = time.perf_counter()
# nodes = [x for x in range(n_nodes)]
# dsat = DSatur(nodes, edges)
# dsat.solve()
# end = time.perf_counter()
# cost_ref = dsat.cost
# solution_ref = [n.color.index for n in dsat.N]
# print(f"DSatur executed in {end - start:.4f} seconds")
