from itertools import product
from random import choice
import random
from operator import itemgetter
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform
from openset_imagenet.util import get_sets_for_ensemble, get_binary_output_for_class_per_model
"""
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
"""

def hamming_distance_min_among_all(matrix, row=True):
    if row:
        # Compute pairwise Hamming distances between rows
        # pdist returns proportion, so we multiply by the number of columns
        hamming_dist_matrix = squareform(pdist(matrix, 'hamming')) * matrix.shape[1]
    else:
        # Transpose the matrix for column-wise comparisons
        # pdist returns proportion, so we multiply by the number of rows
        hamming_dist_matrix = squareform(pdist(matrix.T, 'hamming')) * matrix.shape[0]

    # set the diagonal to infinity since we don't want to consider distance of a row/column to itself
    np.fill_diagonal(hamming_dist_matrix, np.inf)

    # Find the minimum non-zero distance for each row/column
    min_distances = np.min(hamming_dist_matrix, axis=1)

    # Return the overall minimum
    return np.min(min_distances)

def get_sets_for_ensemble_optimized(number_of_models, classes):
    number_of_classes = len(classes)
    # Create a matrix containing all possible combinations of 0 and 1
    combinations = list(product([0, 1], repeat=number_of_classes))
    combinations = np.array(combinations)
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

        # get the index of the maximum minimum_column_wise_distance
        max_min_column_wise_distance = max(minimum_column_wise_distance)
        max_indices = [i for i, x in enumerate(minimum_column_wise_distance) if x == max_min_column_wise_distance]

        if len(max_indices) > 1:
            # if there are multiple maximum values, get the maximum of the minimum_row_wise_distance
            max_min_row_wise_distance = max([minimum_row_wise_distance[i] for i in max_indices])
            max_indices = [i for i in max_indices if minimum_row_wise_distance[i] == max_min_row_wise_distance]

        if len(max_indices) > 1:
            # if there are still multiple indices, choose one randomly
            random_max_sum_index = random.choice(max_indices)
        else:
            random_max_sum_index = max_indices[0]
       
        matrix = np.vstack((matrix, balanced_combinations[random_max_sum_index]))
        # remove the selected vector from the list of balanced combinations
        balanced_combinations.pop(random_max_sum_index)
        print("Matrix shape: ", matrix.shape)
    
    print("Matrix: \n", matrix)
    print("Matrix shape: ", matrix.shape)
    print("Row wise min hamming distance - new algo: ", hamming_distance_min_among_all(matrix, row=True))
    print("Column wise min hamming distance - new algo: ", hamming_distance_min_among_all(matrix, row=False))
    
    # create class splits from the matrix
    class_splits = []
    for i in range(number_of_models):
        # zip the classes with the corresponding row in the matrix
        index_and_class = list(zip(classes, matrix[i]))
        split_0 = [x[0] for x in index_and_class if x[1] == 0]
        split_1 = [x[0] for x in index_and_class if x[1] == 1]
        # save the splits in dictionary
        class_splits.append({0: split_0, 1: split_1})

    return class_splits

get_sets_for_ensemble_optimized(number_of_models=4, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# Old method returns no matrices directly thats why we have to do some extra steps before calculating hamming distance

number_of_models = 4
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_splits = get_sets_for_ensemble(classes, number_of_models)
class_binary = get_binary_output_for_class_per_model(class_splits)

# Convert the dictionary to a list of tuples
class_binary_tuples = list(class_binary.items())
# Sort the tuples by the keys
class_binary_tuples.sort(key=lambda x: x[0])

# Create a numpy array from the sorted list of tuples
class_binary_array = np.array([value for _, value in class_binary_tuples]).T
print(class_binary_array)


print("Row wise min hamming distance - old algo: ", hamming_distance_min_among_all(class_binary_array, row=True))
print("Column wise min hamming distance - old algo: ", hamming_distance_min_among_all(class_binary_array, row=False))