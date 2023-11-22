from collections import namedtuple

Item = namedtuple("Item", ["index", "value", "weight"])


def parse_data(input_data):
    # read all lines from text file
    with open(input_data, "r") as input_data_file:
        input_data = input_data_file.read()
    # split data on newlines
    split_data = input_data.split("\n")
    # parse first line
    firstline = split_data.pop(0).split()
    item_count = int(firstline[0])
    capacity = int(firstline[1])
    # parse remaining lines
    split_data_clean = [x.split() for x in split_data if x != ""]
    items = [Item(i, int(x[0]), int(x[1])) for i, x in enumerate(split_data_clean)]
    return (item_count, capacity, items)


def greedy_knapsack(item_count, capacity, items):
    value_densities = [item.value / item.weight for item in items]
    zipped = list(zip(items, value_densities))
    zipped.sort(reverse=True, key=lambda x: (x[1], x[0].index))
    sorted_items = [x[0] for x in zipped]

    knapsack_value = 0
    knapsack_capacity = capacity
    knapsack_index = []

    for item in sorted_items:
        if item.weight <= knapsack_capacity:
            knapsack_index.append(item.index)
            knapsack_value += item.value
            knapsack_capacity -= item.weight
            if knapsack_capacity == 0:
                break
    return knapsack_index


def report_knapsack(item_count, capacity, items, knapsack_index):
    knapsack_solution = [0] * item_count
    for i in knapsack_index:
        knapsack_solution[i] = 1

    knapsack_items = [items[i] for i in knapsack_index]
    knapsack_items.sort(key=lambda x: x.index)
    knapsack_value = sum([item.value for item in knapsack_items])
    knapsack_weight = sum([item.weight for item in knapsack_items])

    print(50 * "-")
    print("Solution:")
    print(item_count, capacity)
    print(knapsack_solution)
    print(50 * "-")
    print("Stats:")
    print(f"{knapsack_value=}")
    print(f"{knapsack_weight=}/{capacity}")
    print(f"knapsack_items={len(knapsack_items)}/{item_count}")
    print(50 * "-")


INPUT_DATA = "src/32.knapsack/data/ks_30_0"
item_count, capacity, items = parse_data(INPUT_DATA)
solution = greedy_knapsack(item_count, capacity, items)
report_knapsack(item_count, capacity, items, solution)
