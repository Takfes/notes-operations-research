import math
from itertools import combinations

import numpy as np


def Himmelblau(x, y):
    """
    Calculates the value of the Himmelblau function at the given coordinates (x, y).
    see : https://en.wikipedia.org/wiki/Himmelblau%27s_function

    The Himmelblau function is a multi-modal function that has multiple local minima and maxima.
    It is commonly used as a benchmark function for optimization algorithms.

    Parameters:
    - x (float): The x-coordinate.
    - y (float): The y-coordinate.

    Returns:
    - float: The value of the Himmelblau function at the given coordinates.
    """
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def calculate_chromosome_length(range_min, range_max, precision):
    """
    Calculates the length of a chromosome based on the given range and precision.

    Parameters:
    range_min (float): The minimum value of the range.
    range_max (float): The maximum value of the range.
    precision (float): The precision of the chromosome.

    Returns:
    int: The length of the chromosome.

    """
    N = (range_max - range_min) / precision + 1
    L = math.ceil(math.log2(N))
    return L


def binary_to_integer(binary_segment, verbose=False):
    """
    Converts a binary segment to its corresponding integer value.

    Parameters:
        binary_segment (str): The binary segment to be converted.
        verbose (bool, optional): If True, the conversion process will be displayed step by step.
        Defaults to False.

    Returns:
        int: The integer value of the binary segment.
    """

    if verbose:
        iv = 0
        for i, x in enumerate(reversed(binary_segment)):
            print(f"Counting backwards, digit {i} = {x}")
            print(f"{x} * 2^{i} = {2**i}")
            iv += int(x) * 2 ** (i)
            print(f"Running sum : {iv=}")
        return iv
    else:
        return int(binary_segment, 2)


def binary_to_real(binary_segment, range_min, range_max):
    """
    Converts a binary segment to a real value within a specified range.

    Parameters:
    binary_segment (str): The binary segment to be converted.
    range_min (float): The minimum value of the desired range.
    range_max (float): The maximum value of the desired range.

    Returns:
    float: The real value representation of the binary segment within the specified range.
    """
    integer_value = binary_to_integer(binary_segment)
    real_value = range_min + (integer_value / (2 ** len(binary_segment) - 1)) * (
        range_max - range_min
    )
    return real_value


def create_chromosome_segment(chromosome_length):
    """
    Generate a random chromosome segment of the specified length.

    Parameters:
    chromosome_length (int): The length of the chromosome segment.

    Returns:
    str: A string representing the randomly generated chromosome segment.
    """
    return "".join(map(str, np.random.randint(0, 2, chromosome_length)))


def evaluate_chromosome(chromosome, length_per_variable, range_min, range_max):
    """
    Evaluates a chromosome by converting it to real values and calculating the value of the Himmelblau function.

    Parameters:
    chromosome (str): The chromosome to be evaluated.
    length_per_variable (int): The length of the chromosome segment for each variable.
    range_min (float): The minimum value of the range.
    range_max (float): The maximum value of the range.

    Returns:
    float: The value of the Himmelblau function at the real values represented by the chromosome.
    """
    x_chromosome = chromosome[:length_per_variable]
    y_chromosome = chromosome[length_per_variable:]
    x_real = binary_to_real(x_chromosome, range_min, range_max)
    y_real = binary_to_real(y_chromosome, range_min, range_max)
    return Himmelblau(x_real, y_real), x_real, y_real


def tournament_operation(population, tournament_rounds, total_chromosome_length):
    """
    Selects parents using a tournament selection process.

    Args:
        population (int): The population size of randomly generated chromosomes.
        tournament_rounds (int): The number of tournament rounds to be performed. Best chromosome is selected from each round.

    Returns:
        list: The selected parents.

    """
    parents = []
    for _ in range(tournament_rounds):
        population = [
            create_chromosome_segment(total_chromosome_length)
            for _ in range(POPULATION)
        ]
        evaluated = [
            evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
            for x in population
        ]
        best_index = np.array(evaluated).argmin()
        parents.append(population[best_index])
    return parents


def crossover_operation(
    chromosome1,
    chromosome2,
    length_per_variable,
    digits_to_change=2,
    crossover_probability=1.0,
):
    """
    Perform crossover operation on two chromosomes.

    Args:
        chromosome1 (str): The first chromosome.
        chromosome2 (str): The second chromosome.
        length_per_variable (int): The length of each variable in the chromosomes.
        digits_to_change (int, optional): The number of digits to change during crossover. Defaults to 2.
        crossover_probability (float, optional): The probability of performing crossover. Defaults to 1.0.

    Returns:
        tuple: A tuple containing two child chromosomes resulting from crossover.
    """
    if np.random.rand() >= crossover_probability:
        return chromosome1, chromosome2

    idx1, idx2, idx3 = (
        length_per_variable - digits_to_change,
        length_per_variable,
        length_per_variable * 2 - digits_to_change,
    )
    child1 = (
        chromosome1[:idx1]
        + chromosome2[idx1:idx2]
        + chromosome1[idx2:idx3]
        + chromosome2[idx3:]
    )
    child2 = (
        chromosome2[:idx1]
        + chromosome1[idx1:idx2]
        + chromosome2[idx2:idx3]
        + chromosome1[idx3:]
    )
    assert len(child1) == len(chromosome1) == len(chromosome2) == len(child2)
    return child1, child2


def mutation_operation(chromosome, mutation_probability=0.1):
    """
    Performs mutation operation on the given chromosome.

    Args:
        chromosome (str): The chromosome to be mutated.
        mutation_probability (float, optional): The probability of mutation for each gene in the chromosome.
            Defaults to 0.1.

    Returns:
        str: The mutated chromosome.
    """
    mutation_matrix = [
        (np.random.rand() < mutation_probability) * 1 for _ in range(len(chromosome))
    ]
    return "".join([str(int(x) ^ y) for x, y in zip(chromosome, mutation_matrix)])


def evaluate_operation(chomosomes):
    scores = [
        evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
        for x in chomosomes
    ]
    print(f"best : {min(scores):.8f} out of {len(scores)} ")


# Example usage for the Himmelblau function
Himmelblau(3, 2)
Himmelblau(-2.805118, 3.131312)
Himmelblau(-3.779310, -3.283186)
Himmelblau(3.584428, -1.848126)

# For the Himmelblau function
range_min, range_max = -5, 5  # Range for the Himmelblau function variables
precision = 0.01

# Define Chromosome length
length_per_variable = calculate_chromosome_length(range_min, range_max, precision)
print(f"Length per variable: {length_per_variable}")

total_chromosome_length = (
    length_per_variable * 2
)  # since there are two variables x and y
print(f"Total chromosome length: {total_chromosome_length}")

x_chromosome = create_chromosome_segment(length_per_variable)
y_chromosome = create_chromosome_segment(length_per_variable)
xy_chromosome = x_chromosome + y_chromosome
print(f"Chromosome: {xy_chromosome} of lentgh {len(xy_chromosome)}")

# Example usage for a chromosome segment
x_real = binary_to_real(x_chromosome, range_min, range_max)
y_real = binary_to_real(y_chromosome, range_min, range_max)
print(
    f"x chromosome: {x_chromosome} lentgh {len(x_chromosome)} | Real value : {x_real:.6f}"
)
print(
    f"y chromosome: {y_chromosome} lentgh {len(y_chromosome)} | Real value : {y_real:.6f}"
)

# Himmelblau function value
Himmelblau(x_real, y_real)

# 1. Tournament Selection
TOURNAMENT_ROUNDS = 50
POPULATION = 1000

tournament_parents = tournament_operation(
    population=POPULATION, tournament_rounds=TOURNAMENT_ROUNDS
)
eval_tournament_parents = [
    evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
    for x in tournament_parents
]

print(
    f"TS best : {min(eval_tournament_parents):.8f} out of {len(eval_tournament_parents)} "
)

# 2. Cross-Overs between TS results
crossover_childs = set()
combs = list(combinations(tournament_parents, 2))
for i, comb in enumerate(combs):
    child1, child2 = crossover_operation(comb[0], comb[1], length_per_variable)
    crossover_childs.add(child1)
    crossover_childs.add(child2)

crossover_childs = list(crossover_childs)

eval_crossover_childs = [
    evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
    for x in crossover_childs
]

print(
    f"CO best : {min(eval_crossover_childs):.8f} out of {len(eval_crossover_childs)} "
)

# 3. Mutation Operation
mutated_childs = [mutation_operation(x, 0.1) for x in crossover_childs]
eval_mutated_childs = [
    evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
    for x in mutated_childs
]

print(f"MO best : {min(eval_mutated_childs):.8f} out of {len(eval_mutated_childs)} ")

# 4. Selection of the best chromosomes
consolidated_chromosomes = list(
    set(tournament_parents + crossover_childs + mutated_childs)
)
eval_consolidated_chromosomes = [
    evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
    for x in consolidated_chromosomes
]
evaluate_generation = zip(consolidated_chromosomes, eval_consolidated_chromosomes)
evaluate_generation = sorted(evaluate_generation, key=lambda x: x[1])
next_generation = [x[0] for x in evaluate_generation[:TOURNAMENT_ROUNDS]]


# 5. Repeat
def genetic_algorithm(
    population_size,
    tournament_rounds,
    crossover_probability,
    mutation_probability,
    generations,
    range_min,
    range_max,
    precision,
):
    length_per_variable = calculate_chromosome_length(range_min, range_max, precision)
    total_chromosome_length = length_per_variable * 2
    parents = tournament_operation(
        population_size, tournament_rounds, total_chromosome_length
    )
    evaluate_operation(parents)
    for _ in range(generations):
        children = set()
        combs = list(combinations(parents, 2))
        for i, comb in enumerate(combs):
            child1, child2 = crossover_operation(
                comb[0], comb[1], length_per_variable, crossover_probability
            )
            children.add(child1)
            children.add(child2)
        children = list(children)
        mutated_children = [
            mutation_operation(x, mutation_probability) for x in children
        ]
        consolidated_chromosomes = list(set(parents + children + mutated_children))
        eval_consolidated_chromosomes = [
            evaluate_chromosome(x, length_per_variable, range_min, range_max)[0]
            for x in consolidated_chromosomes
        ]
        evaluate_generation = zip(
            consolidated_chromosomes, eval_consolidated_chromosomes
        )
        evaluate_generation = sorted(evaluate_generation, key=lambda x: x[1])
        next_generation = [x[0] for x in evaluate_generation[:tournament_rounds]]
        parents = next_generation
    return next_generation


ga = genetic_algorithm(
    population_size=500,
    tournament_rounds=50,
    crossover_probability=1.0,
    mutation_probability=0.2,
    generations=10,
    range_min=-5,
    range_max=5,
    precision=0.01,
)
