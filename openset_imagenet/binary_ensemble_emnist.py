import random
import time
import sys
import pathlib
from collections import OrderedDict, defaultdict
import numpy
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device
import vast
from loguru import logger
from .metrics import confidence_binary, auc_score_binary, auc_score_multiclass
from .dataset_emnist import Dataset_EMNIST
from .model import LeNet5, load_checkpoint, save_checkpoint, set_seeds
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
import tqdm

def get_class_from_label(label, class_dict):
    """ Get the class from the label.

    Args:
        label (int): Label
        class_dict (dict): Dictionary with the class splits
    Returns:
        torch.float32: Class
    """
    # Check which class the label belongs to and replace the label with that class
    for key, value in class_dict.items():
        if label in value:
            if key != 0 and key != 1:
                print(key)
                raise ValueError("The class label is not 0 or 1")
            found_label = torch.as_tensor(key, dtype=torch.float32)
            return found_label
    return None

def train_ensemble(model, data_loader, class_dict, optimizer, loss_fn, trackers, cfg):
    """ Main training loop.

    Args:
        model (torch.model): Model
        data_loader (torch.DataLoader): DataLoader
        optimizer (torch optimizer): optimizer
        loss_fn: Loss function
        trackers: Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    j = None

    # training loop
    if not cfg.parallel:
        data_loader = tqdm.tqdm(data_loader)
    for images, labels in data_loader:
        # Check which class the label belongs to and replace the label with that class
        for i in range(len(labels)):
            label = labels[i].item()
            # replace the label with the class 0 or 1
            labels[i] = get_class_from_label(label, class_dict)
        
        model.train()  # To collect batch-norm statistics set model to train mode
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad()
        images = device(images)
        labels = device(labels)

        # Forward pass
        logits, features = model(images)

        # Calculate loss
        targets = labels.view(-1, 1)
        targets = targets.type(torch.float32)
        j = loss_fn(logits, targets)
        trackers["j"].update(j.item(), batch_len)
        # Backward pass
        j.backward()
        optimizer.step()


def validate_ensemble(model, data_loader, class_dict, loss_fn, n_classes, trackers, cfg):
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
    else:
        min_unk_score = 1. / n_classes
        unknown_class = -1
        last_valid_class = None


    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, n_classes), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, features = model(images)
            scores = torch.nn.functional.sigmoid(logits) #TODO changed from softmax to sigmoid for bce

            # we need scores to be either 0 or 1
            threshold = 0.5

            # Apply thresholding to get binary values 0 or 1
            scores = (scores >= threshold).type(torch.int64)
            
             # get the class from the label either 0 or 1
            for index in range(len(labels)):
                label = labels[index].item()
                # replace the label with the class 0 or 1
                labels[index] = get_class_from_label(label, class_dict)

            # targets = labels.view(-1,)
            targets = labels.type(torch.float32)
            j = loss_fn(logits, targets.view(-1, 1))
            trackers["j"].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            start_ix = i * cfg.batch_size # i does not have to be = 1
            all_targets[start_ix: start_ix + batch_len] = targets
            all_scores[start_ix: start_ix + batch_len] = scores
        
        # show difference between all_scores and all_targets
        print(len(all_scores), len(all_targets))
        print("The Tensors match in number of cases: ", torch.eq(all_scores.view(-1,), all_targets.view(-1,)).sum())

        kn_conf, kn_count, neg_conf, neg_count = confidence_binary(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score, # TODO change this to 0.5 for bce?
            unknown_class = unknown_class,
            last_valid_class = last_valid_class)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


def get_arrays(model, loader, garbage, pretty=False):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
        garbage (bool): Whether to remove final logit value
    """
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model.logits.out_features  # logits output classes
        if garbage:
            logits_dim -= 1
        features_dim = model.logits.in_features  # features dimensionality
        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, logits_dim), device="cpu")

        index = 0
        if pretty:
            loader = tqdm.tqdm(loader)
        for images, labels in loader:
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logits, feature = model(images)
            # compute softmax scores
            scores = torch.nn.functional.softmax(logits, dim=1)
            # shall we remove the logits of the unknown class?
            # We do this AFTER computing softmax, of course.
            if garbage:
                logits = logits[:,:-1]
                scores = scores[:,:-1]
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logits.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = scores.detach().cpu()
            index += curr_b_size
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())

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
    # print(unique_classes, type(unique_classes))
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
    print("Ensemble training class splits: ", class_splits)
    return class_splits



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
    
    # create datasets
    # train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
    # val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))

    train_ds = Dataset_EMNIST(
        dataset_root=cfg.data.dataset_path,
        which_set="train",
        include_unknown=False,  #TODO: change this to True for bce
        has_garbage_class=False)
    
    val_ds = Dataset_EMNIST(
        dataset_root=cfg.data.dataset_path,
        which_set="validation",
        include_unknown=False,  #TODO: change this to True for bce
        has_garbage_class=False)

        # If using garbage class, replaces label -1 to maximum label + 1
    if cfg.loss.type == "garbage":
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
    elif cfg.loss.type == "softmax": #TODO remove negatives from training and validation set for bce too?
            # remove the negative label from softmax training set, not from val set!
            train_ds.remove_negative_label()
    

    # Create unique class splits for ensemble set-vs-set training
    print(train_ds.unique_classes, cfg.algorithm.num_models)
    class_splits = get_sets_for_ensemble(train_ds.unique_classes, cfg.algorithm.num_models)

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

    # Set dictionaries to keep track of the losses for each model
    t_metrics = [defaultdict(AverageMeter) for _ in range(cfg.algorithm.num_models)]
    v_metrics = [defaultdict(AverageMeter) for _ in range(cfg.algorithm.num_models)]

    # set loss
    loss = None
    if cfg.loss.type == "entropic":
        # number of classes - 1 since we have no label for unknown
        n_classes = train_ds.label_count - 1
    elif cfg.loss.type == "bce":
        # one probability only because we can model the prob for negative class as 1-prob
        n_classes = 1 #TODO: check if this is correct or not and we need 2
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
    # bce loss for ensemble
    elif cfg.loss.type == "bce":
        loss = torch.nn.BCEWithLogitsLoss()


    # Create the models and save them in a
    models = []
    for i in range(cfg.algorithm.num_models):
        model = LeNet5(fc_layer_dim=84,
                         out_features=n_classes,
                         logit_bias=False)
        device(model)
        models.append(model)

    # Create optimizer and scheduler for each model
    opts = []
    schedulers = []
    for model in models:
        if cfg.opt.type == "sgd":
            opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
        else:
            opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)
        opts.append(opt)

        # Learning rate scheduler
        if cfg.opt.decay > 0:
            scheduler = lr_scheduler.StepLR(
                opt,
                step_size=cfg.opt.decay,
                gamma=cfg.opt.gamma,
                verbose=True)
        else:
            scheduler = None
        schedulers.append(scheduler)

    # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        START_EPOCH, BEST_SCORE = load_checkpoint(
            model=models[0],
            checkpoint=cfg.checkpoint,
            opt=opts[0],
            scheduler=schedulers[0])
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")

    # Print info to console and setup summary writer
    # Info on console
    logger.info("============ Data ============")
    logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"workers: {cfg.workers}")
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"optimizer: {cfg.opt.type}")
    logger.info(f"Learning rate: {cfg.opt.lr}")
    logger.info(f"Device: {cfg.gpu}")
    logger.info(f"number of models: {cfg.algorithm.num_models}")
    logger.info("Training...")
    writer = SummaryWriter(log_dir=cfg.output_directory, filename_suffix="-"+cfg.log_name)
    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()
        # training loop
        for model, opt, class_split, t_metric in zip(models, opts, class_splits, t_metrics):
            train_ensemble(
                model=model,
                data_loader=train_loader,
                class_dict=class_split,
                optimizer=opt,
                loss_fn=loss,
                trackers=t_metric,
                cfg=cfg)
        train_time = time.time() - epoch_time

        # validation loop
        for model, class_split, v_metric in zip(models, class_splits, v_metrics):
            validate_ensemble(
                model=model,
                data_loader=val_loader,
                class_dict=class_split,
                loss_fn=loss,
                n_classes=n_classes,
                trackers=v_metric,
                cfg=cfg)
        curr_score = v_metric["conf_kn"].avg + v_metric["conf_unk"].avg
        # learning rate scheduler step
        for scheduler in schedulers:
            if scheduler is not None:
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
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")
        # save best model and current model
        ckpt_name = cfg.model_path.format(cfg.output_directory, cfg.loss.type, "threshold", "curr")
        save_checkpoint(ckpt_name, models[0], epoch, opts[0], curr_score, scheduler=schedulers[0])
        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = cfg.model_path.format(cfg.output_directory, cfg.loss.type, "threshold", "best")
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model {ckpt_name} at epoch: {epoch}")
            save_checkpoint(ckpt_name, models[0], epoch, opts[0], BEST_SCORE, scheduler=schedulers[0])
        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break
    # clean everything
    del models
    logger.info("Training finished")
