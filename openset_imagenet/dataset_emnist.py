import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)


class Dataset_EMNIST(torch.utils.data.dataset.Dataset):
    """A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the first 13 letters will be used as negatives (for training and validation), and the last 13 letters will serve as unknowns (for testing only).
    The MNIST test set is used both in the validation and test split of this dataset.

    For the test set, you should consider to leave the parameters `include_unknown` and `has_garbage_class` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    has_garbage_class: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.
    """
    def __init__(self, dataset_root, which_set="train", include_unknown=True, has_garbage_class=False):
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.which_set = which_set
        targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
        print("targets: ", targets)
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.has_garbage_class = has_garbage_class

        self.unique_classes = np.sort(np.unique(self.mnist.targets.tolist()))
        if has_garbage_class:
            self.unique_classes = np.append(self.unique_classes, 10)
        if include_unknown:
            self.unique_classes = np.append(self.unique_classes, -1)
        self.label_count = len(self.unique_classes)


    def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index]
        else:
            return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 10 if self.has_garbage_class else -1

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes)