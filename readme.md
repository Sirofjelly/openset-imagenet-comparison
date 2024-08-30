# Open-Set Classification on ImageNet using an Ensemble of Binary Classifiers
Implementation of the experiments performed in the master thesis Open-Set Classification with Ensembles of Binary Classifiers from Silvan Kuebler. 
If you make use of our binary classifier approaches implementation, please cite the master thesis as follows:

    @masterthesis{kuebler2024binaryclassifiers,
        author       = {Kuebler, Silvan},
        title        = {Open-Set Classification with Ensembles of Binary Classifiers},
        year         = {2024},
        school = {University of Zurich}
    }

If you make use of the ImageNet Protocols, please cite the the work as follows:

    @inproceedings{palechor2023openset,
        author       = {Palechor, Andres and Bhoumik, Annesha and G\"unther, Manuel},
        booktitle    = {Winter Conference on Applications of Computer Vision (WACV)},
        title        = {Large-Scale Open-Set Classification Protocols for {ImageNet}},
        year         = {2023},
        organization = {IEEE/CVF}
    }

These experiments have been extended to include more algorithms, including MaxLogits, OpenMax, EVM and PROSER.
The publication is currently under review:

    @article{bisgin2023large,
        title     = {Large-Scale Evaluation of Open-Set Image Classification Techniques},
        author    = {Bisgin, Halil and Palechor, Andres and Suter, Mike and G\"unther, Manuel},
        journal   = {\textbf{under submission}},
        year      = {2023}
    }



## LICENSE
This code package is open-source based on the BSD license.
Please see `LICENSE` for details.

## Data

The scripts rely on the ImageNet dataset using the ILSVRC 2012 data and the EMNIST dataset.
If you do not have a copy yet the ImageNet dataset can be [downloaded from Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview) (untested). The EMNIST dataset can be [downloaded from Kaggle](https://www.kaggle.com/datasets/crawford/emnist) (untested).
The protocols rely on the `robustness` library, which in turn relies on some files that have been distributed with the ImageNet dataset some time ago, but they are not available anymore.
With a bit of luck, you can find the files somewhere online:

* imagenet_class_index.json
* wordnet.is_a.txt
* words.txt

If not, you can also rely on the pre-computed protocol files, which can be found in the provided `protocols.zip` file and extracted via:

    unzip protocols.zip


## Setup

We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate openset-imagenet-comparison

## Scripts

The directory `openset_imagenet/script` includes several scripts, which are automatically installed and runnable.

### Protocols

You can generate the protocol files using the command `imagenet_protocols.py`.
Please refer to its help for details:

    protocols_imagenet.py --help

Basically, you have to provide the original directory for your ImageNet images, and the directory containing the files for the `robustness` library.
The other options should be changed rarely.

### Training of one base model

The training can be performed using the `train.py` script.
It relies on a configuration file as can be found in `config/binary_ensemble.yaml`.
Please set all parameters as required (the default values are as used in the paper), and run:

    train.py [config] [protocol] -g GPU

where `[config]` is the configuration file, `[protocol]` one of the three protocols.
The `-g` option can be used to specify that the training should be performed on the GPU (**highly recommended**), and you can also specify a GPU index in case you have several GPUs at your disposal.

### Evaluation

In order to evaluate all models on the test sets, you can make use of the `evaluate_binary_ensemble.py` script. The evaluation method needs to be specified using `--threshold`, the loss must be selected using `--losses`, the protocols using `--protocols`, the model which was used using `--algorithms`, the configuration file path `--configuration`.

### Plotting

Finally, the `plot_all.py` script can be used to create the oscr plots from the master thesis.

## Getting help

In case of trouble, feel free to contact me under [silvan.kuebler@uzh.ch](mailto:silvan.kuebler@uzh.ch?subject=Open-Set%20Binary%20Classifiers)
