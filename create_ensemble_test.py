import numpy as np
from itertools import product
from random import choice
import random
from operator import itemgetter
import itertools
from openset_imagenet.util import get_sets_for_ensemble, get_binary_output_for_class_per_model

def hamming_distance_min_among_all(matrix, row=True):
    if row == True:
        # Calculate the Hamming distance between rows in the matrix and for every row get the minimum distance to other rows
        hamming_distances = []
        for i in range(matrix.shape[0]):
            row = matrix[i]
            distances = []
            for j in range(matrix.shape[0]):
                if i != j:
                    # Calculate the Hamming distance between two rows
                    distance = np.sum(row != matrix[j])
                    distances.append(distance)
            # Get the minimum distance for the current row
            hamming_distances.append(min(distances))
        # Return the minimum distance among all rows
        return min(hamming_distances)
    else:
        # Calculate the Hamming distance between columns in the matrix and for every column get the minimum distance to other columns
        hamming_distances = []
        for i in range(matrix.shape[1]):
            column = matrix[:, i]
            distances = []
            for j in range(matrix.shape[1]):
                if i != j:
                    distance = np.sum(column != matrix[:, j])
                    distances.append(distance)
            hamming_distances.append(min(distances))
        return min(hamming_distances)
        

def create_matrix(number_of_models, number_of_classes):
    # Create a matrix containing all possible combinations of 0 and 1
    combinations = list(product([0, 1], repeat=number_of_classes))
    combinations = np.array(combinations)
    print("Combinations: \n", combinations)
    print("Number of Combinations: ", len(combinations))  # should be 2^number_of_classes

    # remove vectors that begin with 1 to avoid inversion
    combinations = [c for c in combinations if c[0] == 0]

    # check which are balanced vectors
    balanced_combinations = []
    for combination in combinations:
        if (number_of_classes % 2 == 0):
            balanced = np.sum(combination) == number_of_classes // 2
        else:
            balanced = np.logical_or(np.sum(combination) == number_of_classes // 2, np.sum(combination) == (number_of_classes + 1) // 2)
        if np.all(balanced):
            balanced_combinations.append(combination)
    print("Balanced Combinations: ", balanced_combinations, "Length (Max possible Models): ", len(balanced_combinations))

    if len(balanced_combinations) < number_of_models:
        raise ValueError("Number of balanced combinations is less than number of models")

    # randomly select the first vector to start the matrix
    first_vector_index = random.randint(0, len(balanced_combinations) - 1)
    matrix = balanced_combinations[first_vector_index]
    # remove the start vector from the list of balanced combinations
    balanced_combinations.pop(first_vector_index)

    # calculate the optimal next vector to add to the matrix
    for _ in range(number_of_models - 1):
        minimum_row_wise_distance = []
        minimum_column_wise_distance = []
        for balanced_combination in balanced_combinations:
            matrix = np.vstack((matrix, balanced_combination))
            minimum_row_wise_distance.append(hamming_distance_min_among_all(matrix, row=True))
            minimum_column_wise_distance.append(hamming_distance_min_among_all(matrix, row=False))
            # remove the last row from the matrix
            matrix = matrix[:-1]
        # get the index of the minimum distance
        row_dist_and_column_dist = list(zip(minimum_row_wise_distance, minimum_column_wise_distance))
        # find the tuple with the maximum sum
        max_sum_tuple = max(row_dist_and_column_dist, key=lambda x: x[0] + x[1])

        # get all possibles index of the maximum sum tuple
        max_sum_index = [i for i, j in enumerate(row_dist_and_column_dist) if j == max_sum_tuple]
        random_max_sum_index = choice(max_sum_index)
        matrix = np.vstack((matrix, balanced_combinations[random_max_sum_index]))
        # remove the selected vector from the list of balanced combinations
        balanced_combinations.pop(random_max_sum_index)
    
    print("Matrix: \n", matrix)
    print("Matrix shape: ", matrix.shape)
    print("row wise min hamming distance: ", hamming_distance_min_among_all(matrix, row=True))
    print("column wise min hamming distance: ", hamming_distance_min_among_all(matrix, row=False))
    return matrix

create_matrix(100, 10)

"""
# Old method returns no matrices directly thats why we have to do some extra steps before calculating hamming distance

number_of_models = 6
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
"""