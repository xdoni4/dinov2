# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import json
import time
import math
import torch
import torch.nn as nn
import logging
import argparse
import dinov2.distributed as distributed


from functools import partial
from torch.amp import GradScaler
from fvcore.common.checkpoint import PeriodicCheckpointer
from torchmetrics.classification import Accuracy, F1Score, AUROC, Precision, Recall


from dinov2.utils.config import setup
from dinov2.logging import MetricLogger
from dinov2.fsdp import FSDPCheckpointer
from dinov2.fsdp import reshard_fsdp_model
from dinov2.utils.utils import CosineScheduler
from dinov2.data import (
    DataAugmentationDINO3D,
    DataAugmentation3DForClassification,
    DataAugmentation3DForClassificationVal,
    collate_data_and_cast,
    collate_data_for_test,
    InfinitePrefetchedDataloader
)
from dinov2.data.augmentations import (
    DataAugmentationDINO3DOpenmind,
    DataAugmentation3DForClassificationOpenmind,
    DataAugmentation3DForClassificationValOpenmind
)
from dinov2.data.dataset import (
    AbdomenAtlasPreprocessed, 
    AMOSCTUnlabeledTrainPreprocessed,
    CTRATETrainPreprocessed,
    CTRATEValPreprocessed,
    OpenNeuroPreprocessed,
    MedicalImageDatasetDINO,
    BraTSClassificationTrain,
    BraTSClassificationVal
)
from dinov2.data.masking import MaskingGenerator3D
from dinov2.train.ssl_meta_arch import SSLMetaArch
from nnssl.data.raw_dataset import Collection
from nnssl.ssl_data.dataloading.utils import get_subject_identifiers

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration, only_save=False):
    # test_data_train_transform = DataAugmentation3DForClassification(device=torch.cuda.current_device(), dtype=torch.float16)
    test_data_train_transform = DataAugmentation3DForClassificationOpenmind(device=torch.cuda.current_device(), dtype=torch.float16)
    # test_data_val_transform = DataAugmentation3DForClassificationVal(device=torch.cuda.current_device(), dtype=torch.float16)
    test_data_val_transform = DataAugmentation3DForClassificationValOpenmind(device=torch.cuda.current_device(), dtype=torch.float16)
    test_collate_fn = collate_data_for_test
    
   

    def create_test_dataloader_train():
        test_dataset_prefetch = MedicalImageDatasetDINO(
            sources = [BraTSClassificationTrain()], # CTRATETrainPreprocessed()
            prefetch=True,
            replication=32,
            prefetch_workers=4,
            prefetch_buffer_size=128,
            fields=['image', 'labels']
        )
        dl = InfinitePrefetchedDataloader(
            test_dataset_prefetch,
            cfg.evaluation.per_device_batch_size,
            test_collate_fn,
            test_data_train_transform
        )
        logger.info("Warming up test train data prefetcher.")
        time.sleep(30)
        logger.info("Warming up completed.")
        return dl

    def create_test_dataloader_val():
        test_dataset_val = MedicalImageDatasetDINO(
            sources = [BraTSClassificationVal()], # CTRATEValPreprocessed()
            prefetch=True,
            replication=1,
            prefetch_buffer_size=128,
            fields=['image', 'labels']
        )

        dl = InfinitePrefetchedDataloader(
            test_dataset_val,
            cfg.evaluation.per_device_batch_size,
            test_collate_fn,
            test_data_val_transform
        )
        logger.info("Warming up test val data prefetcher.")
        time.sleep(30)
        logger.info("Warming up completed.")
        return dl

    if cfg.evaluation.get("debug", None):
        new_teacher_state_dict = model.teacher.state_dict()
        new_cls_head_state_dict = model.cls_model.head.state_dict()    
        if distributed.is_main_process():
            iterstring = str(iteration)
            eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
            os.makedirs(eval_dir, exist_ok=True)
            # save teacher checkpoint
            teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint_before.pth")
            cls_head_ckp_path = os.path.join(eval_dir, "cls_head_checkpoint_before.pth")
            torch.save({"teacher": new_teacher_state_dict}, teacher_ckp_path)
            torch.save({"cls_head" : new_cls_head_state_dict}, cls_head_ckp_path)
    
    if not only_save:
        model.cls_model.backbone.eval()
        test_data_loader_train = create_test_dataloader_train()

        fp16_scaler = GradScaler()

        logger.info("Starting linear probing validation epoch.")

        train_metric_logger = MetricLogger(delimiter="  ", output_file=None)
        val_metric_logger = MetricLogger(delimiter="  ", output_file=None)
        header_train = "Linear probing train"
        header_val = "Linear probing val"
        start_iter = 0
        max_iter = 20 * len(test_data_loader_train.prefetched_dataset) // (test_data_loader_train.batch_size * distributed.get_global_size())
        if cfg.evaluation.get("debug", None):
            max_iter = 200
        cur_iteration = start_iter
        optimizer = torch.optim.AdamW(model.cls_model.head.parameters(), lr=cfg.evaluation.lr)

        for data in train_metric_logger.log_every(
            test_data_loader_train,
            10,
            header_train,
            max_iter,
            start_iter,
        ):
            if cur_iteration > max_iter:
                break

            images, labels = data
            current_batch_size = len(images)

            # compute losses
            optimizer.zero_grad(set_to_none=True)
            pred = model.cls_model(images)
            reshard_fsdp_model(model.cls_model)

            loss = nn.BCEWithLogitsLoss()(pred, labels)
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

            train_metric_logger.update(lr=cfg.evaluation.lr)
            train_metric_logger.update(current_batch_size_val=current_batch_size)
            train_metric_logger.update(total_loss_val=loss.detach().item())

            cur_iteration = cur_iteration + 1
        del data
        test_data_loader_train.prefetched_dataset.destroy()
        del test_data_loader_train

        test_data_loader_val = create_test_dataloader_val()

        accuracy = Accuracy(task='MULTILABEL', num_labels=model.cls_model.head.n_classes, average='macro').to(torch.cuda.current_device())
        f1_macro = F1Score(task='MULTILABEL', num_labels=model.cls_model.head.n_classes, average='macro').to(torch.cuda.current_device())
        f1_weighted = F1Score(task='MULTILABEL', num_labels=model.cls_model.head.n_classes, average='weighted').to(torch.cuda.current_device())
        recall = Recall(task='multilabel', num_labels=model.cls_model.head.n_classes, average='macro').to(torch.cuda.current_device())
        precision = Precision(task='multilabel', num_labels=model.cls_model.head.n_classes, average='macro').to(torch.cuda.current_device())
        auroc = AUROC(task='multilabel', num_labels=model.cls_model.head.n_classes, average='macro').to(torch.cuda.current_device())
        auroc_all = AUROC(task='multilabel', num_labels=model.cls_model.head.n_classes, average='none').to(torch.cuda.current_device())

        max_iter = len(test_data_loader_val.prefetched_dataset) // (test_data_loader_val.batch_size * distributed.get_global_size())
        if cfg.evaluation.get("debug", None):
            max_iter = 10
        for data in val_metric_logger.log_every(
            test_data_loader_val,
            10,
            header_val,
            n_iterations=max_iter
        ):
            images, labels = data
            with torch.no_grad():
                pred = model.cls_model(images)
                reshard_fsdp_model(model.cls_model)
                accuracy.update(pred, labels)
                f1_macro.update(pred, labels)
                f1_weighted.update(pred, labels)
                recall.update(pred, labels)
                precision.update(pred, labels)
                auroc.update(pred, labels.int())
                auroc_all.update(pred, labels.int())

        del data
        test_data_loader_val.prefetched_dataset.destroy()
        del test_data_loader_val

    new_teacher_state_dict = model.teacher.state_dict()
    new_cls_head_state_dict = model.cls_model.head.state_dict()    
    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        cls_head_ckp_path = os.path.join(eval_dir, "cls_head_checkpoint.pth")
        torch.save({"teacher": new_teacher_state_dict}, teacher_ckp_path)
        torch.save({"cls_head" : new_cls_head_state_dict}, cls_head_ckp_path)
    
    if only_save:
        return {}

    return_dict = {
        "validation_accuracy" : accuracy.compute(), 
        "validation_f1_macro" : f1_macro.compute(),
        "validation_f1_weighted" : f1_weighted.compute(),
        "validation_recall" : recall.compute(),
        "validation_precision" : precision.compute(),
        "validation_auroc" : auroc.compute(),
    }
    auroc_all_computed = auroc_all.compute()
    auroc_all_dict = {f"validation_auroc_class_{i}" : x for i, x in enumerate(auroc_all_computed)}
    return_dict.update(auroc_all_dict)
    
    train_metric_logger.synchronize_between_processes()
    return_dict.update({k: meter.global_avg for k, meter in train_metric_logger.meters.items()})
    return return_dict


def do_train(cfg, model, resume=False):
    model.train()
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    mask_generator = MaskingGenerator3D(
        input_size=(128 // 16, 128 // 16, 128 // 16),
        max_num_patches=0.5 * 128 // 16 * 128 // 16 * 128 // 16
    )

    data_transform = DataAugmentationDINO3DOpenmind(
        local_crops_number=4,
        initial_crop_size=140,
        global_crops_size=128,
        local_crops_size=64,
        batch_size=1,
        device=torch.cuda.current_device(),
        dtype=torch.float16
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.2, 0.5),
        mask_probability=0.2,
        n_tokens=(128 // 16) ** 3,
        mask_generator=mask_generator,
        dtype=torch.float16,
        labels_in_sample=False
    )

    def create_train_dataloader():
        os.environ['nnssl_raw'] = '/home/jovyan/datasets/OpenMind'
        os.environ['nnssl_preprocessed'] = '/home/jovyan/misha/misc/cotomka'

        with open("/home/jovyan/misha/misc/cotomka/Dataset745_OpenMind/pretrain_data__onemmiso.json", "r") as f:
            pretrain_data_json = json.load(f)

        collection = Collection.from_dict(pretrain_data_json)

        subject_identifiers = get_subject_identifiers("/home/jovyan/misha/misc/cotomka/Dataset745_OpenMind/nnsslPlans_onemmiso/")
        dataset_openneuro = OpenNeuroPreprocessed(
            "/home/jovyan/misha/misc/cotomka/Dataset745_OpenMind/nnsslPlans_onemmiso/",
            collection,
            subject_identifiers=subject_identifiers
        )

        dataset_prefetch = MedicalImageDatasetDINO(
            # sources = [CTRATETrainPreprocessed(), AbdomenAtlasPreprocessed(), AMOSCTUnlabeledTrainPreprocessed()],
            sources = [dataset_openneuro],
            transform=None,
            prefetch=True,
            replication=32,
            prefetch_workers=8,
            prefetch_buffer_size=256,
            fields=['image']
        )

        dl = InfinitePrefetchedDataloader(
            dataset_prefetch,
            cfg.train.batch_size_per_gpu,
            collate_fn,
            data_transform
        )
        logger.info("Warming up train data prefetcher.")
        time.sleep(30)
        logger.info("Warming up completed.")
        return dl

    data_loader = create_train_dataloader()

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] // 2
        if iteration > max_iter:
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0: #
            del data
            data_loader.prefetched_dataset.destroy()

            only_save = False
            validation_metrics = do_test(cfg, model, f"manual_{iteration}", only_save=only_save)
            if not only_save:
                metric_logger.update(**validation_metrics)
            torch.cuda.synchronize()
            
            dl = create_train_dataloader()
            data_loader.prefetched_dataset = dl.prefetched_dataset
            model.train()

        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
