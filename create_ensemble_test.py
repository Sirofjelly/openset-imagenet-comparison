import numpy as np
from itertools import product
from random import choice
import random
import itertools
from openset_imagenet.util import get_sets_for_ensemble, get_binary_output_for_class_per_model

def create_matrix(number_of_classes, number_of_models):
    # Create a matrix containing all possible combinations of 0 and 1
    combinations = list(product([0, 1], repeat=number_of_models))
    combinations = np.array(combinations)

    # Generate all possible matrices based on the randomly selected first vector
    matrices = []
    first_vector = choice(combinations)
    print("First Vector: ", first_vector)
    remaining_combinations = [c for c in combinations if not np.array_equal(c, first_vector)]

    for remaining_vectors in itertools.combinations(remaining_combinations, number_of_classes - 1):
        remaining_vectors = np.array(remaining_vectors)
        matrix = np.column_stack((first_vector, remaining_vectors.T))
        matrices.append(matrix)
    print(len(matrices))

    # Filter out matrices with unbalanced 0 and 1 per row
    balanced_matrices = []
    for matrix in matrices:
        row_sums = matrix.sum(axis=1)
        if (number_of_classes % 2 == 0):
            balanced = row_sums == number_of_classes // 2
        else:
            balanced = np.logical_or(row_sums == number_of_classes // 2, row_sums == (number_of_classes + 1) // 2)
        if np.all(balanced):
            balanced_matrices.append(matrix)
    print("Balanced Matrixes: ", len(balanced_matrices))

    # Filter out matrices with inverted rows and duplicate rows
    unique_matrices = []
    for matrix in balanced_matrices:
        matrix_list = list(matrix)
        inversed_rows_exist = False

        # check if a row is duplicated in matrix
        unk, count = np.unique(matrix, axis=0, return_counts=True)
        if any(count > 1):
            continue
        
        for row in matrix:
            inverted = np.ones(len(row), dtype=np.int8) - row

            if any(np.array_equal(inverted, m) for m in matrix_list):
                inversed_rows_exist = True
                break
            
        if not inversed_rows_exist:
            unique_matrices.append(matrix)
    print("No inversed rows and no duplicate rows: ", len(unique_matrices))
    print(unique_matrices)

    # Calculate the total Hamming distance between columns for each matrix
    hamming_distances = []
    for matrix in unique_matrices:
        total_hamming_distance = 0
        for i in range(number_of_classes):
            for j in range(i + 1, number_of_classes):
                total_hamming_distance += np.sum(matrix[:, i] != matrix[:, j])
        hamming_distances.append(total_hamming_distance)

    # Return the matrices and their corresponding Hamming distances
    print("Hamming distance new approach: ", set(hamming_distances))
    return unique_matrices, hamming_distances

# New method
create_matrix(4, 3)


# Old method returns no matrices directly thats why we have to do some extra steps before calculating hamming distance

number_of_models = 3
classes = [0, 1, 2, 3]
class_splits = get_sets_for_ensemble(classes, number_of_models)
class_binary = get_binary_output_for_class_per_model(class_splits)
print("class binaries: ", class_binary)

# Convert the dictionary to a list of tuples
class_binary_tuples = list(class_binary.items())
# Sort the tuples by the keys
class_binary_tuples.sort(key=lambda x: x[0])

# Create a numpy array from the sorted list of tuples
class_binary_array = np.array([value for _, value in class_binary_tuples]).T


hamming_distances = []
total_hamming_distance = 0
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        total_hamming_distance += np.sum(class_binary_array[:, i] != class_binary_array[:, j])
hamming_distances.append(total_hamming_distance)

# Return the matrices and their corresponding Hamming distances
print("Hamming distance old approach: ", set(hamming_distances))
