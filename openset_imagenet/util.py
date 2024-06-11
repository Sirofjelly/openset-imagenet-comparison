"""Set of utility functions to produce evaluation figures and histograms."""

from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib import colors
import matplotlib.cm as cm
import random
import numpy
import torch
from itertools import product
from scipy.spatial.distance import pdist, squareform
import sys

import yaml

class NameSpace:
    def __init__(self, config):
        # recurse through config
        config = {name : NameSpace(value) if isinstance(value, dict) else value for name, value in config.items()}
        self.__dict__.update(config)

    def __repr__(self):
        return "\n".join(k+": " + str(v) for k,v in vars(self).items())

    def __getitem__(self, key):
        # get nested NameSpace by key
        return vars(self)[key]

    def dump(self, indent=4):
        return yaml.dump(self.dict(), indent=indent)

    def dict(self):
        return {k: v.dict() if isinstance(v, NameSpace) else v for k,v in vars(self).items()}

def load_yaml(yaml_file):
    """Loads a YAML file into a nested namespace object"""
    config = yaml.safe_load(open(yaml_file, 'r'))
    return NameSpace(config)



def dataset_info(protocol_data_dir):
    """ Produces data frame with basic info about the dataset. The data dir must contain train.csv, validation.csv
    and test.csv, that list the samples for each split.
    Args:
        protocol_data_dir: Data directory.
    Returns:
        data frame: contains the basic information of the dataset.
    """
    data_dir = Path(protocol_data_dir)
    files = {'train': data_dir / 'train.csv', 'val': data_dir / 'validation.csv',
             'test': data_dir / 'test.csv'}
    pd.options.display.float_format = '{:.1f}%'.format
    data = []
    for split, path in files.items():
        df = pd.read_csv(path, header=None)
        size = len(df)
        kn_size = (df[1] >= 0).sum()
        kn_ratio = 100 * kn_size / len(df)
        kn_unk_size = (df[1] == -1).sum()
        kn_unk_ratio = 100 * kn_unk_size / len(df)
        unk_unk_size = (df[1] == -2).sum()
        unk_unk_ratio = 100 * unk_unk_size / len(df)
        num_classes = len(df[1].unique())
        row = (split, num_classes, size, kn_size, kn_ratio, kn_unk_size,
               kn_unk_ratio, unk_unk_size, unk_unk_ratio)
        data.append(row)
    info = pd.DataFrame(data, columns=['split', 'classes', 'size', 'kn size', 'kn (%)', 'kn_unk size',
                                       'kn_unk (%)', 'unk_unk size', 'unk_unk (%)'])
    return info


def read_array_list(file_names):
    """ Loads npz saved arrays
    Args:
        file_names: dictionary or list of arrays
    Returns:
        Dictionary of arrays containing logits, scores, target label and features norms.
    """
    list_paths = file_names
    arrays = defaultdict(dict)

    if isinstance(file_names, dict):
        for key, file in file_names.items():
            arrays[key] = np.load(file)
    else:
        for file in list_paths:
            file = str(file)
            name = file.split('/')[-1][:-8]
            arrays[name] = np.load(file)
    return arrays


def calculate_oscr(gt, scores, unk_label=-1):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    # get predicted class for known samples
    pred_class = np.argmax(scores, axis=1)[kn]
    correctly_predicted = pred_class == gt[kn]
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    # get maximum scores for unknown samples
    max_score = np.max(scores, axis=1)[unk]

    # Any max score can be a threshold
    thresholds = np.unique(max_score)

    #print(target_score) #HB
    for tau in thresholds:
        # compute CCR value
        val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
        ccr.append(val)

        val = (max_score >= tau).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def ccr_at_fpr(gt, scores, fpr_values, unk_label=-1):

    # compute ccr and fpr values from scores
    ccr, fpr = calculate_oscr(gt, scores, unk_label)

    ccrs = []
    for t in fpr_values:
        # get the FPR value that is closest, but above the current threshold
        candidates = np.nonzero(np.maximum(t - fpr, 0))[0]
        if candidates.size > 0:
            ccrs.append(ccr[candidates[0]])
        else:
            ccrs.append(None)

    return ccrs


# get distinguishable colors
import matplotlib.cm
colors = matplotlib.cm.tab10(range(10))

COLORS = {
    "threshold": colors[1],
    "binary_ensemble_combined_imagenet": colors[2],
    "openmax": colors[8],
    "proser": colors[2],
    "evm": colors[3],
    "maxlogits": colors[5],
}

STYLES = {
    "entropic": "dashed",
    "softmax": "solid",
    "bce": "solid",
    "bce_neg": "dashed",
    "garbage": "dotted",
    "p1": "dashed",
    "p2": "dotted",
    "p3": "solid"
}

NAMES = {
    "threshold": "MSS",
    "entropic": "EOS",
    "bce": "BCE",
    "bce_neg": "BCE-Neg",
    "binary_ensemble_combined_imagenet": "Binary Ensemble",
    "openmax": "OpenMax",
    "proser": "PROSER*",
    "evm": "EVM",
    "maxlogits": "MLS",
    "binary_ensemble_emnist": "BE",
    "softmax": "Softmax",
    "garbage": "Garbage",
    "p1": "P_1",
    "p2": "P_2",
    "p3": "P_3",
    1: "$P_1$",
    2: "$P_2$",
    3: "$P_3$"
}

def plot_single_oscr(fpr, ccr, ax, loss, algorithm, scale, line_style=None):
    linewidth = 1.1
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Manual limits
        ax.set_ylim(0.09, 1)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif scale == 'semilog':
        ax.set_xscale('log')
        # Manual limits
        ax.set_ylim(0.0, .8)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # MaxNLocator(7))  #, prune='lower'))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_ylim(0.0, 0.8)
        # ax.set_xlim(None, None)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # , prune='lower'))
    # Remove fpr=0 since it cause errors with different ccrs and logscale.
#    if len(x):
#        non_zero = x != 0
#        x = x[non_zero]
#        y = y[non_zero]
    ax.plot(fpr,
            ccr,
            linestyle=line_style or STYLES[loss],
            color=COLORS[algorithm],
            linewidth=linewidth)  # marker='2', markersize=1
    return ax


def plot_oscr(arrays, gt, scale='linear', title=None, ax_label_font=13, ax=None, unk_label=-1, line_style=None):
    """Plots OSCR curves for all given scores.
    The scores are stored as arrays: Float array of dim [N_samples, N_classes].
    The arrays contain scores for various loss functions and algorithms as arrays[loss][algorithm].
    """

    for loss, loss_arrays in arrays.items():
        for algorithm, scores in loss_arrays.items():
            ccr, fpr = calculate_oscr(gt, scores, unk_label)
            ax = plot_single_oscr(fpr, ccr,
                              ax=ax,
                              loss=loss,
                              algorithm=algorithm,
                              scale=scale,
                              line_style=line_style)
    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)

    return ax

def oscr_legend(losses, algorithms, figure, **kwargs):
    """Creates a legend with the different line style and colors"""
    # add dummy plots for the different styles
    from matplotlib.lines import Line2D

    # create legend elements
    if len(losses) > 1:
        empty_legend = Line2D([None], [None], marker=".", visible=False)
        padding = len(algorithms) - len(losses)
        a_padding = max(-padding,0)
        l_padding = max(padding, 0)

        # add legend elements with sufficient padding
        legend_elements = \
                [empty_legend]*(l_padding//2) + \
                [Line2D([None], [None], linestyle=STYLES[loss], color="k") for loss in losses] + \
                [empty_legend]*(l_padding//2 + l_padding%2) + \
                [empty_legend]*(a_padding//2) + \
                [Line2D([None], [None], linestyle="solid", color=COLORS[algorithm]) for algorithm in algorithms] + \
                [empty_legend]*(a_padding//2 + + a_padding%2)

        labels = \
                [""] *(l_padding//2) + \
                [NAMES[loss] for loss in losses] + \
                [""]*(l_padding//2 + l_padding%2) + \
                [""] *(a_padding//2) + \
                [NAMES[algorithm] for algorithm in algorithms] + \
                [""]*(a_padding//2 + + a_padding%2)

        # re-order row-first to column-first
        columns = max(len(algorithms), len(losses))

        indexes = [i for j in range(columns) for i in (j, j+columns)]
        legend_elements = [legend_elements[index] for index in indexes]
        labels = [labels[index] for index in indexes]

    else:
        legend_elements = \
                [Line2D([None], [None], linestyle="solid", color=COLORS[algorithm]) for algorithm in algorithms]

        labels = \
                [NAMES[algorithm] for algorithm in algorithms]

        columns = len(algorithms)


    figure.legend(handles=legend_elements, labels=labels, loc="lower center", ncol=columns, **kwargs)



def get_histogram(scores,
                  gt,
                  bins=100,
                  log_space=False,
                  geomspace_limits=(1, 1e2)):
    """Calculates histograms of scores"""
    known = gt >= 0
    unknown = gt == -2
    negative = gt == -1

    knowns = scores[known, gt[known]]
    unknowns = np.amax(scores[unknown], axis=1)
    negatives = np.amax(scores[negative], axis=1)

    if log_space:
        lower, upper = geomspace_limits
        bins = np.geomspace(lower, upper, num=bins)
#    else:
#        bins = np.linspace(0, 1, num=bins+1)
    histograms = {}
    histograms["known"] = np.histogram(knowns, bins=bins)
    histograms["unknown"] = np.histogram(unknowns, bins=bins)
    histograms["negative"] = np.histogram(negatives, bins=bins)
    return histograms


# new util functions
def get_binary_output_for_class_per_model(class_splits):
    """ Get the binary class representation for each class."""
    all_classes = []
    all_classes = class_splits[0][0] + class_splits[0][1]   
    all_classes.sort()

    # get the binary class representation
    class_binary = {}
    for c in all_classes:
        binary_code = []
        for class_split in class_splits:
            if c in class_split[0]:
                binary_code.append(0)
            elif c in class_split[1]:
                binary_code.append(1)
            else:
                raise ValueError("Class not found in any split")
        class_binary[c] = binary_code
    return class_binary

def get_similarity_score_from_binary_to_label(model_binary, class_binary):
    """
    Get the predicted class from the binary sigmoid output of the model. The lower the similarity the worse. Exact match is == num_models
    Args:
        model_binary (list): Binary output of the model.
        class_binary (dict): Binary representation of the classes.
    """
    num_outputs = len(model_binary)
    model_binary = model_binary.cpu()
    model_binary = model_binary.view(-1,)

    # get the class from the binary output
    class_similarities = numpy.empty(len(class_binary))
    for i, (c, b) in enumerate(class_binary.items()):
        similarity =  numpy.sum(numpy.abs(numpy.array(b) - numpy.array(model_binary)))
        class_similarities[i] = num_outputs - similarity
    return torch.from_numpy(class_similarities)

def get_similarity_score_from_binary_to_label_new(model_binary, class_binary):
    """
    Used when using logits output from the model without sigmoid.
    Get the predicted class from the binary output of the model. The lower the similarity the worse. Exact match is == num_models
    Args:
        model_outputs (list): Binary output of the model.
        class_binary (dict): Binary representation of the classes.
    """
    num_outputs = len(model_binary)
    model_binary = model_binary.cpu()
    model_binary = model_binary.view(-1,)

    # get the class from the binary output
    class_similarities = numpy.empty(len(class_binary))
    for i, (c, b) in enumerate(class_binary.items()):
        b = numpy.array(b) * 2 - 1 # convert to -1 and 1
        similarity =  numpy.sum((b * numpy.array(model_binary)))
        class_similarities[i] = similarity
    return torch.from_numpy(class_similarities)

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

def get_sets_for_ensemble_hamming(number_of_models, classes):
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
    print("Row wise min hamming distance - hamming algo: ", hamming_distance_min_among_all(matrix, row=True))
    print("Column wise min hamming distance - hamming algo: ", hamming_distance_min_among_all(matrix, row=False))
    
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


def get_sets_for_ensemble(unique_classes, num_models):
    """ Create the splits for the ensemble training.
    Args:
        unique_classes (list): List of unique classes
        num_models (int): Number of models in the ensemble
    Returns:
        list: List of dictionaries with the class splits
    """
    class_splits = []
    shuffled_classes = []
    split_size = len(unique_classes) // 2
    unique_classes = list(unique_classes)

    for i in range(num_models):
        # check if we had the same shuffle before or the exact opposite
        while True:
            classes = random.sample(unique_classes, len(unique_classes))
            split_0 = classes[:split_size]
            split_1 = classes[split_size:]
            split_0.sort()
            split_1.sort()
            if (split_0, split_1) not in shuffled_classes and (split_1, split_0) not in shuffled_classes:
                shuffled_classes.append((split_0, split_1))
                # finally add the split to the list
                class_splits.append({0: split_0, 1: split_1})
                break
            else:
                print("this split does already exist: ", split_1, split_0)

    # we check if each class is unique if not we rerun the function
    class_binary = get_binary_output_for_class_per_model(class_splits)
    class_binary_tuples = list(class_binary.items())
    # Sort the tuples by the keys
    class_binary_tuples.sort(key=lambda x: x[0])

    # Create a numpy array from the sorted list of tuples
    class_binary_array = np.array([value for _, value in class_binary_tuples]).T
    column_ham_dist = hamming_distance_min_among_all(class_binary_array, row=False)
    if column_ham_dist == 0:
        print("Two or more columns are the same, rerun the function")
        return get_sets_for_ensemble(unique_classes, num_models)
    
    print("Ensemble training class splits: ", class_splits)
    return class_splits

def get_class_from_label(label, class_dict, unknown_in_both=False, unknown_for_training=False):
    """ Get the class from the label.
        Based on the input class it searches for for the class and returns the label.

    Args:
        label (int): Label
        class_dict (dict): Dictionary with the class splits
        separate_class_for_unknown (bool): If True, the unknown class is separated from the rest. And we do have 3 classes.
    Returns:
        torch.float32: Class, either 0 or 1
    """
    # Check which class the label belongs to and replace the label with that class
    for key, value in class_dict.items():
        if label == -1 and unknown_in_both:
            # we want to have probability 0.5 for both
            return torch.as_tensor(0.5, dtype=torch.float32)
        
        elif label == -1 and not unknown_for_training:
            return torch.as_tensor(-1, dtype=torch.float32)
        
        if label in value:
            if key != 0 and key != 1:
                print(key)
                raise ValueError("The class label is not 0 or 1")
            found_label = torch.as_tensor(key, dtype=torch.float32)
            return found_label
    return None
