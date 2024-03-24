""" Independent code for inference in testing dataset."""
import argparse
import os, sys
from pathlib import Path
import numpy as np
from openset_imagenet.model import EnsembleModel
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import openset_imagenet
import pickle
from openset_imagenet.openmax_evm import compute_adjust_probs, compute_probs, get_param_string
from loguru import logger

def command_line_options(command_line_arguments=None):
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--threshold",
        choices = ["threshold", "logits"],
        default = ["threshold"],
        help="Which evaluation criteria to use. If True, the threshold is used. If False, the logits distance is used."
    )
    # directory parameters
    parser.add_argument(
        "--losses", "-l",
        choices = ["entropic", "softmax", "garbage", "bce"],
        nargs="+",
        default = ["entropic", "softmax", "garbage"],
        help="Which loss functions to evaluate"
    )
    parser.add_argument(
        "--protocols", "-p",
        type = int,
        nargs = "+",
        choices = (1,2,3),
        default = (2,1,3),
        help = "Which protocols to evaluate"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["threshold", "openmax", "proser", "evm", "binary_ensemble", "binary_ensemble_emnist"],
        nargs = "+",
        default = ["threshold", "openmax", "proser", "evm"],
        help = "Which algorithm to evaluate. Specific parameters should be in the yaml file"
    )

    parser.add_argument(
        "--configuration", "-c",
        type = Path,
        default = Path("config/test.yaml"),
        help = "The configuration file that defines the experiment"
    )

    parser.add_argument(
        "--use-best", "-b",
        action="store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )

    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help = "If selected, all results will always be recomputed. If not selected, already existing results will be skipped."
    )

    args = parser.parse_args(command_line_arguments)
    return args

def dataset(cfg, protocol):
    # Create transformations
    transform = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    if cfg.algorithm.type == "binary_ensemble_emnist":
        test_dataset = openset_imagenet.Dataset_EMNIST(
            dataset_root=cfg.data.dataset_path,
            which_set="test",
            include_unknown=True
        )
    else:
        # We only need test data here, since we assume that parameters have been selected
        test_dataset = openset_imagenet.ImagenetDataset(
            csv_file=cfg.data.test_file.format(protocol),
            imagenet_path=cfg.data.imagenet_path,
            transform=transform)

    # Info on console
    logger.info(f"Loaded test dataset for protocol {protocol} with len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    # return test loader
    return test_dataset, test_loader


def load_model(cfg, loss, algorithm, protocol, suffix, output_directory, n_classes, model_nr=None):
    if algorithm == 'proser':
        opt = cfg.optimized[algorithm]
        popt = opt[f"p{protocol}"][loss]
        base = openset_imagenet.ResNet50(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

        model = openset_imagenet.model.ResNet50Proser(
            dummy_count = popt.dummy_count,
            fc_layer_dim=n_classes,
            resnet_base = base,
            loss_type=loss)

        model_path = opt.output_model_path.format(output_directory, loss, algorithm, popt.epochs, popt.dummy_count, suffix)
    elif cfg.algorithm.type == "binary_ensemble_emnist":
        model = openset_imagenet.LeNet5(
            fc_layer_dim=84,
            out_features=n_classes,
            logit_bias=False)

        model_path = cfg.model_path.format(output_directory, loss, cfg.algorithm.type, suffix, model_nr)
        print(model_path)
    else:
        model = openset_imagenet.ResNet50(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

        model_path = cfg.model_path.format(output_directory, loss, "threshold", suffix)


    if not os.path.exists(model_path):
        logger.warning(f"Could not load model file {model_path}; skipping")
        return

    if cfg.algorithm.type == "binary_ensemble_emnist":
        start_epoch, best_score = openset_imagenet.binary_ensemble_emnist.load_checkpoint(model, model_path)
    elif cfg.algorithm.type == "binary_ensemble":
            start_epoch, best_score = openset_imagenet.binary_ensemble.load_checkpoint(model, model_path)
    else: # all other models
        start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, model_path)

    logger.info(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)
    return model


def extract(model, data_loader, algorithm, loss, threshold):
    if algorithm == 'proser':
         return openset_imagenet.proser.get_arrays(
            model=model,
            loader=data_loader,
            pretty=True
        )
    else:
        return openset_imagenet.binary_ensemble_emnist.get_arrays(
            model=model,
            loader=data_loader,
            garbage=loss=="garbage",
            pretty=True,
            threshold=threshold
        )


def post_process(gt, logits, features, scores, cfg, protocol, loss, algorithm, output_directory, gpu):
    if algorithm in ("threshold", "proser", "binary_ensemble_emnist"):
        print("shape gt", gt.shape)
        print("shape logits", logits.shape)
        print("shape features", features.shape)
        print("shape scores", scores.shape)
        return scores

    opt = cfg.optimized[algorithm]
    popt = opt[f"p{protocol}"][loss]

    if algorithm == "openmax":
        key = get_param_string("openmax", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier)
    else:
        key = get_param_string("evm", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier, cover_threshold=opt.cover_threshold)
    model_path = opt.output_model_path.format(output_directory, loss, algorithm, key, opt.distance_metric)
    if not os.path.exists(model_path):
        logger.warning(f"Could not load model file {model_path}; skipping")
        return
    model_dict = pickle.load(open(model_path, "rb"))

    hyperparams = openset_imagenet.util.NameSpace(dict(distance_metric = opt.distance_metric))
    if algorithm == 'openmax':
        #scores are being adjusted her through openmax alpha
        logger.info("adjusting probabilities for openmax with alpha")
        return compute_adjust_probs(gt, logits, features, scores, model_dict, "openmax", gpu, hyperparams, popt.alpha)
    elif algorithm == 'evm':
        logger.info("computing probabilities for evm")
        return compute_probs(gt, logits, features, scores, model_dict, "evm", gpu, hyperparams)

def write_scores(gt, logits, features, scores, loss, algorithm, suffix, output_directory):
    file_path = Path(output_directory) / f"{loss}_{algorithm}_test_arr_{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    logger.info(f"Target labels, logits, features and scores saved in: {file_path}")

def load_scores(loss, algorithm, suffix, output_directory):
    file_path = Path(output_directory) / f"{loss}_{algorithm}_test_arr_{suffix}.npz"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None


def process_model(protocol, loss, algorithms, cfg, suffix, gpu, force, threshold):
    output_directory = Path(cfg.output_directory)/f"Protocol_{protocol}"

    # set device
    if gpu is not None:
        set_device_gpu(index=gpu)
    else:
        logger.warning("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    # get dataset
    test_dataset, test_loader = dataset(cfg, protocol)

    # load base model
    if loss == "garbage":
        n_classes = test_dataset.label_count - 1 # we use one class for the negatives; the dataset has two additional  labels: negative and unknown
    elif loss == "bce":
        n_classes = 1 # TODO check this, needed for output of the model
    else:
        n_classes = test_dataset.label_count - 2  # number of classes - 2 when training was without garbage class

    if any(a!="proser" for a in algorithms):
        if any(a == "binary_ensemble_emnist" for a in algorithms): #TODO Binary ensemble for ImageNet
                base_data = None if force else load_scores(loss, cfg.algorithm.type, suffix, output_directory)
                if base_data is None: # when we never executed this before
                    logger.info(f"Loading base models for protocol {protocol}, {loss}")
                    # load base model
                    base_models = torch.nn.ModuleList()
                    for i in range(cfg.algorithm.num_models):
                        base_model = load_model(cfg, loss, cfg.algorithm.type, protocol, suffix, output_directory, n_classes, model_nr=i)
                        base_models.append(base_model)
                    # create ensemble
                    base_model = EnsembleModel(base_models)
                    if base_model is not None:
                        # extract features
                        logger.info(f"Extracting base scores for protocol {protocol}, {loss}")
                        gt, logits, features, base_scores = extract(base_model, test_loader, cfg.algorithm.type, loss, threshold)
                        write_scores(gt, logits, features, base_scores, loss, cfg.algorithm.type, suffix, output_directory)
                        # remove model from GPU memory
                        del base_model
                else:
                    logger.info("Using previously computed features")
                    gt, logits, features, base_scores = base_data["gt"], base_data["logits"], base_data["features"], base_data["scores"]
        else: # when we do not need for loop
            base_data = None if force else load_scores(loss, "threshold", suffix, output_directory)
            if base_data is None:
                logger.info(f"Loading base model for protocol {protocol}, {loss}")
                # load base model
                base_model = load_model(cfg, loss, "threshold", protocol, suffix, output_directory, n_classes)
                if base_model is not None:
                    # extract features
                    logger.info(f"Extracting base scores for protocol {protocol}, {loss}")
                    targets, logits, features, base_scores = extract(base_model, test_loader, "threshold", loss)
                    write_scores(targets, logits, features, base_scores, loss, "threshold", suffix, output_directory)
                    # remove model from GPU memory
                    del base_model
            else:
                logger.info("Using previously computed features")
                gt, logits, features, base_scores = base_data["gt"], base_data["logits"], base_data["features"], base_data["scores"]

    for algorithm in algorithms:
        if algorithm not in ("proser", "threshold", "binary_ensemble_emnist"):
            logger.info(f"Post-processing scores for protocol {protocol}, {loss} with {algorithm}")
            # post-process scores
            scores = post_process(gt, logits, features, base_scores, cfg, protocol, loss, algorithm, output_directory, gpu)
            if scores is not None:
                write_scores(gt, logits, features, scores, loss, algorithm, suffix, output_directory)
        if algorithm == "proser":
            proser_data = None if force else load_scores(loss, "proser", suffix, output_directory)
            if proser_data is None:
                # load proser model
                logger.info(f"Loading proser model for protocol {protocol}, {loss}")
                proser_model = load_model(cfg, loss, "proser", protocol, suffix, output_directory, n_classes)
                if proser_model is not None:
                    # and extract features using that model
                    logger.info(f"Extracting proser scores for protocol {protocol}, {loss}")
                    proser_gt, proser_logits, proser_features, proser_scores = extract(proser_model, test_loader, "proser", loss)
                    write_scores(proser_gt, proser_logits, proser_features, proser_scores, loss, "proser", suffix, output_directory)
                    del proser_model
            else:
                logger.info("Relying on previously defined proser scores")
    logger.info(f"Finished processing protocol {protocol}, {loss}")



def main(command_line_arguments = None):

    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)
    suffix = "best" if args.use_best else "curr"

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')
    for protocol in args.protocols:
        for loss in args.losses:
            process_model(protocol, loss, args.algorithms, cfg, suffix, args.gpu, args.force, args.threshold)

if __name__=='__main__':
    main()
