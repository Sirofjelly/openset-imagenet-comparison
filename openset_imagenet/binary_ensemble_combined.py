import random
import time
import sys
import pathlib
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import functools
from torchvision import ops
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device
import vast
from loguru import logger
from .metrics import confidence_combined_binary, auc_score_binary, auc_score_multiclass, confidence
from .dataset import ImagenetDataset
from .dataset_emnist import Dataset_EMNIST
from .model import ResNet50, ResNet50Plus, LeNet5, load_checkpoint, save_checkpoint, set_seeds
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
import tqdm
from os import path
from .util import get_sets_for_ensemble, get_sets_for_ensemble_hamming, hamming_distance_min_among_all, get_class_from_label, get_similarity_score_from_binary_to_label, get_similarity_score_from_binary_to_label_new, get_binary_output_for_class_per_model

def transform_labels(labels, class_dicts, unknown_in_both=False, unknown_for_training=False):
    """
    Transforms the labels to intermediate labels either 0 or 1 for the model to learn its binary classification task.
    Args:
        labels: The original labels
        class_dicts: The class dictionaries for each model
        unknown_in_both: Whether the unknown class is in both models
        unknown_for_training: Whether the unknown class is used for training
    Returns:
        intermediate_labels: The intermediate labels for the model
    """
    intermediate_labels = device(torch.zeros((len(labels), len(class_dicts))))

    for i, class_dict in enumerate(class_dicts):
        intermediate_labels[:, i] = device(torch.tensor([get_class_from_label(label.item(), class_dict, unknown_in_both=unknown_in_both, unknown_for_training=unknown_for_training) for label in labels]))

    return intermediate_labels

def train(model, data_loader, class_dicts, optimizer, loss_fn, trackers, cfg):
    """Main training loop."""
    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    j = None  # training loop
    if not cfg.parallel:
        data_loader = tqdm.tqdm(data_loader)

    for images, labels in data_loader:
        images = device(images)
        labels = device(labels)
        # Check which partition the label belongs to and replace the label with that class either 0 or 1
        intermediate_labels = transform_labels(labels, class_dicts, unknown_in_both=cfg.unknown_in_both)
  
        model.train()  # To collect batch-norm statistics set model to train mode
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad()
        intermediate_labels = device(intermediate_labels)

        # Forward pass
        logits, features = model(images)
        j = loss_fn(logits, intermediate_labels)

        trackers["j"].update(j.item(), batch_len)
        # Backward pass
        j.backward()
        optimizer.step()


def validate(model, data_loader, class_dicts, loss_fn, n_classes, trackers, cfg):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
        loss_fn: Loss function
        n_classes(int): Total number of classes
        trackers(dict): Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset all validation metrics
    for metric in trackers.values():
        metric.reset()

    if cfg.loss.type == "garbage":
        min_unk_score = 0.
        unknown_class = n_classes - 1
        last_valid_class = -1
    elif cfg.loss.type == "bce" or cfg.loss.type == "sigmoid-focal":
        min_unk_score = 0
        unknown_class = -1
        last_valid_class = None
    else:
        min_unk_score = 1. / n_classes
        unknown_class = -1
        last_valid_class = None


    model.eval()
    with torch.no_grad():
        # get class binary for each model
        class_binaries = get_binary_output_for_class_per_model(class_dicts)
        data_len = len(data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, len(class_binaries)), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, features = model(images)
            scores = torch.nn.functional.sigmoid(logits)          

            # we initialize class scores
            class_scores = torch.empty((batch_len, len(class_binaries)))
            # Apply thresholding to get binary values 0 or 1
            for indi in range(scores.shape[0]):
                class_scores[indi, :] = get_similarity_score_from_binary_to_label(model_binary=scores[indi, :], class_binary=class_binaries)
            
             # get the class from the label either 0 or 1
            intermediate_labels = transform_labels(labels, class_dicts, unknown_in_both=cfg.unknown_in_both, unknown_for_training=cfg.unknown_for_training)

            # targets = labels.view(-1,)
            targets = intermediate_labels.type(torch.float32)
            targets = device(targets)
            j = loss_fn(logits, targets)
            trackers["j"].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            start_ix = i * cfg.batch_size # i does not have to be = 1
            all_targets[start_ix: start_ix + batch_len] = labels
            all_scores[start_ix: start_ix + batch_len] = class_scores

        kn_conf, kn_count, neg_conf, neg_count = confidence_combined_binary(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score,
            unknown_class = unknown_class,
            last_valid_class = last_valid_class,
            number_of_binary_outputs=cfg.algorithm.num_models)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


def get_arrays(model, loader, garbage, pretty=False, threshold=True, remove_negative=False, cfg=None):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
        garbage (bool): Whether to remove final logit value
    """
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)    # dataset length
        if cfg.algorithm.model == "resnet50Plus":
            logits_dim = len(model.output_layers)
        else:
            logits_dim = model.logits.out_features  # logits output classes
        if garbage:
            logits_dim -= 1
        
        class_binaries = get_binary_output_for_class_per_model(model.class_split)
        if cfg.algorithm.model == 'resnet50Plus':
            features_dim = len(model.output_layers)
        else:
            features_dim = model.logits.in_features  # features dimensionality
        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, len(class_binaries) -1 if remove_negative else len(class_binaries)), device="cpu")

        index = 0
        if pretty:
            loader = tqdm.tqdm(loader)
        for images, labels in loader:
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logits, feature = model(images)
            # compute softmax scores
            scores_sig = torch.nn.functional.sigmoid(logits)
            final_class_score = torch.empty((curr_b_size, len(class_binaries)), device="cpu")
            if threshold == "threshold":
                scores = (scores_sig >= 0.5).type(torch.int64) # by doing this we lose lots of information
                for i in range(scores.shape[0]): 
                    final_class_score[i, :] = get_similarity_score_from_binary_to_label(model_binary=scores[i, :], class_binary=class_binaries)
            elif threshold == "probabilities":
                for i in range(scores_sig.shape[0]):
                    final_class_score[i, :] = get_similarity_score_from_binary_to_label(model_binary=scores_sig[i, :], class_binary=class_binaries)
            elif threshold == "logits":
                for i in range(logits.shape[0]):
                    final_class_score[i, :] = get_similarity_score_from_binary_to_label_new(model_binary=logits[i, :], class_binary=class_binaries)
            # shall we remove the logits of the unknown class?
            # We do this AFTER computing softmax, of course.
            if garbage:
                logits = logits[:,:-1]
                scores = scores[:,:-1]
            if cfg.unknown_for_training and not cfg.unknown_in_both:
                # we remove the negative class from the final class score we do this only when the model does output an 
                # extra score for the negative class e.g. when training with it not when the unknown is in both classes
                final_class_score = final_class_score[:,1:]
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logits.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = final_class_score.detach().cpu()
            index += curr_b_size
        print("Shapes: ", all_scores.shape, all_targets.shape, all_logits.shape, all_feat.shape)
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())




def worker(cfg):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    set_seeds(cfg.seed)

    BEST_SCORE = 0.0    # Best validation score
    START_EPOCH = 0     # Initial training epoch

    # Configure logger. Log only on first process. Validate only on first process.
    # msg_format = "{time:DD_MM_HH:mm} {message}"
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= pathlib.Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    # Load datasets
    if not cfg.data.dataset == 'emnist':
        # Set image transformations
        train_tr = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

        val_tr = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

        # create datasets
        train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
        val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))

        if train_file.exists() and val_file.exists():
            train_ds = ImagenetDataset(
                csv_file=train_file,
                imagenet_path=cfg.data.imagenet_path,
                transform=train_tr
            )
            val_ds = ImagenetDataset(
                csv_file=val_file,
                imagenet_path=cfg.data.imagenet_path,
                transform=val_tr
            )

            # If using garbage class, replaces label -1 to maximum label + 1
            if cfg.loss.type == "garbage":
                # Only change the unknown label of the training dataset
                train_ds.replace_negative_label()
                val_ds.replace_negative_label()
            elif cfg.loss.type == "softmax":
                # remove the negative label from softmax training set, not from val set! Or when we do not want to use unknowns
                train_ds.remove_negative_label()
            elif not cfg.unknown_for_training:
                train_ds.remove_negative_label()
            print("Label count for train: ", train_ds.label_count, "Label count for val: ", val_ds.label_count)
            print("Unique classes for train: ", train_ds.unique_classes, "Unique classes for val: ", val_ds.unique_classes)
        else:
            raise FileNotFoundError("train/validation file does not exist")
    # Load EMNIST dataset when it is set like this in the config file
    else:
        train_ds = Dataset_EMNIST(
        dataset_root=cfg.data.dataset_path,
        which_set="train",
        include_unknown=cfg.unknown_for_training,
        has_garbage_class=False)
    
        val_ds = Dataset_EMNIST(
            dataset_root=cfg.data.dataset_path,
            which_set="validation",
            include_unknown=cfg.unknown_for_training,
            has_garbage_class=False)
        

    unique_classes = train_ds.unique_classes
    if cfg.unknown_for_training and cfg.unknown_in_both:
        # remove -1 from the class splits as we want the network to learn 0.5 for unknowns
        unique_classes = np.array([x for x in unique_classes if x != -1])
    if cfg.algorithm.sets == "random":
        class_splits = get_sets_for_ensemble(unique_classes, cfg.algorithm.num_models)
        class_binary = get_binary_output_for_class_per_model(class_splits)
        # Convert the dictionary to a list of tuples
        class_binary_tuples = list(class_binary.items())
        # Sort the tuples by the keys
        class_binary_tuples.sort(key=lambda x: x[0])

        # Create a numpy array from the sorted list of tuples
        class_binary_array = np.array([value for _, value in class_binary_tuples]).T
        print("Row wise min hamming distance - random algo: ", hamming_distance_min_among_all(class_binary_array, row=True))
        print("Column wise min hamming distance - random algo: ", hamming_distance_min_among_all(class_binary_array, row=False))
        
    elif cfg.algorithm.sets == "hamming":
        class_splits = get_sets_for_ensemble_hamming(number_of_models=cfg.algorithm.num_models, classes=unique_classes)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,   
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,)

    # setup device
    if cfg.gpu is not None:
        set_device_gpu(index=cfg.gpu)
    else:
        logger.warning("No GPU device selected, training will be extremely slow")
        set_device_cpu()

    # Callbacks
    early_stopping = None
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # set loss
    loss = None
    if cfg.loss.type == "entropic":
        # number of classes - 1 since we have no label for unknown
        n_classes = train_ds.label_count - 1
    elif cfg.loss.type == "bce":
        # one probability only because we can model the prob for negative class as 1-prob
        n_classes = 1
    else:
        # number of classes when training with extra garbage class for unknowns, or when unknowns are removed
        n_classes = train_ds.label_count

    if cfg.loss.type == "entropic":
        # We select entropic loss using the unknown class weights from the config file
        loss = EntropicOpensetLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == "softmax":
        # We need to ignore the index only for validation loss computation
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif cfg.loss.type == "garbage":
        # We use balanced class weights
        class_weights = device(train_ds.calculate_class_weights())
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif cfg.loss.type == "bce":
        # Binary cross entropy loss with logits
        loss = torch.nn.BCEWithLogitsLoss()
    elif cfg.loss.type == "sigmoid-focal":
        # Sigmoid focal loss to handle hard examples
        loss = functools.partial(ops.sigmoid_focal_loss, alpha=-1, gamma=2, reduction="mean")
    

    # Create the model
    if cfg.data.dataset == 'emnist':
        model = LeNet5(fc_layer_dim=84, out_features=cfg.algorithm.num_models, logit_bias=False)
    # If we are on ImageNet dataset and we want to use the ResNet50+ model
    elif cfg.algorithm.model == 'resnet50Plus':
        model = ResNet50Plus(fc_layer_dim=cfg.algorithm.num_models,
                     out_features=cfg.algorithm.num_models, # number of binary outputs
                     logit_bias=False)
        
    # If we are on ImageNet dataset and we want to use the standard ResNet50 model
    else:
        model = ResNet50(fc_layer_dim=cfg.algorithm.num_models,
                     out_features=cfg.algorithm.num_models, # number of binary outputs
                     logit_bias=False)
    device(model)

    # Create optimizer
    if cfg.opt.type == "sgd":
        opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
    elif cfg.opt.type == "adam":
        opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)

    # Learning rate scheduler
    if cfg.opt.decay > 0:
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg.opt.decay,
            gamma=cfg.opt.gamma,
            verbose=True)
    else:
        scheduler = None


    # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        START_EPOCH, BEST_SCORE = load_checkpoint(
            model=model,
            checkpoint=cfg.checkpoint,
            opt=opt,
            scheduler=scheduler)
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")


    # Print info to console and setup summary writer

    # Info on console
    logger.info("============ Data ============")
    logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info(f"Using Negatives for Training: {cfg.unknown_for_training}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"workers: {cfg.workers}")
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"optimizer: {cfg.opt.type}")
    logger.info(f"Learning rate: {cfg.opt.lr}")
    logger.info(f"Device: {cfg.gpu}")
    logger.info(f"Number of binary outputs: {cfg.algorithm.num_models}")
    logger.info(f"Using following approach to generate sets: {cfg.algorithm.sets}")
    logger.info(f"Using Model: {cfg.algorithm.model}")
    logger.info("Training...")
    writer = SummaryWriter(log_dir=cfg.output_directory, filename_suffix="-"+cfg.log_name)

    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()

        # training loop
        train(
            model=model,
            data_loader=train_loader,
            class_dicts=class_splits,
            optimizer=opt,
            loss_fn=loss,
            trackers=t_metrics,
            cfg=cfg)

        train_time = time.time() - epoch_time

        # validation loop
        validate(
            model=model,
            data_loader=val_loader,
            class_dicts=class_splits,
            loss_fn=loss,
            n_classes=n_classes,
            trackers=v_metrics,
            cfg=cfg)

        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        #  training information on console
        # validation+metrics writer+save model time
        val_time = time.time() - train_time - epoch_time
        def pretty_print(d):
            #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
            return dict(d)

        logger.info(
            f"loss:{cfg.loss.type} "
            f"protocol:{cfg.protocol} "
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")

        # save best model and current model
        ckpt_name = cfg.model_path.format(cfg.output_directory, cfg.loss.type, cfg.algorithm.type, "curr")
        save_checkpoint(ckpt_name, model, epoch, opt, curr_score, scheduler=scheduler, class_split=class_splits)

        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = cfg.model_path.format(cfg.output_directory, cfg.loss.type, cfg.algorithm.type, "best")
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model {ckpt_name} at epoch: {epoch}")
            save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler, class_split=class_splits)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    # clean everything
    del model
    logger.info("Training finished")