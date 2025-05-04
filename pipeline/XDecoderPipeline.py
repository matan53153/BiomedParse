# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
import time
import datetime
import json
import os
import re # Import re for directory name cleaning
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators

from trainer.default_trainer import DefaultTrainer

from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog
from detectron2.structures import ImageList

from modeling import build_model
from modeling.utils import get_class_names
from modeling.BaseModel import BaseModel
from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from utilities.distributed import is_main_process
from utilities.constants import COCO_PANOPTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_metadata, hook_switcher, hook_opt
from modeling.criterion.pixel_criterion import PixelCriterion

logger = logging.getLogger(__name__)


class XDecoderPipeline:
    def __init__(self, opt):
        self._opt = opt
        # logger.info(self._opt['RESUME_FROM']) # Original line was causing issues if RESUME_FROM empty
        # Initialize student criterion if a student model is specified
        self.student_criterion = None
        # <<< --- Check if STUDENT key exists before accessing --- >>>
        if self._opt['MODEL'].get('STUDENT') is not None and \
           self._opt['MODEL']['NAME'] in ['student_resnet50_segmentation',
                                          'student_vit_segmentation',
                                          'student_mobilenet_segmentation']:
        # <<< --- End Check --- >>>
            # Define weights and losses for the simple pixel-wise loss
            # You might want to make these configurable later
            student_weight_dict = {'loss_sem_seg_ce': 1.0} # Example weight
            student_losses = ['sem_seg_ce']
            ignore_index = self._opt['MODEL'].get('IGNORE_VALUE', 255) # Use ignore value from config
            self.student_criterion = PixelCriterion(weight_dict=student_weight_dict,
                                                    losses=student_losses,
                                                    ignore_index=ignore_index)
            logger.info("Initialized PixelCriterion for student model.")

    def initialize_model(self):
        model_name = "default"
        model = build_model(self._opt)
        model.train()

        if is_main_process():
            logger.info(model)

        raw_models = {model_name: BaseModel(self._opt, model)}
        return raw_models

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ) -> Union[DataLoader, iterators.CheckpointableIterator]:
        distributed = self._opt['world_size'] > 1
        if is_evaluation:
            if not hasattr(self, 'valid_loader'):
                dataloaders = build_eval_dataloader(self._opt)
                self.valid_loader = dataloaders
            else:
                dataloaders = self.valid_loader
            idx = 0 if dataset_label=='dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
            dataloader = dataloaders[idx]
            self.evaluator = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
        else:
            if not hasattr(self, 'train_loader'):
                dataloader = build_train_dataloader(self._opt)
                self.train_loader = dataloader
                logger.info(f'num of train samples: {len(dataloader)}')
            else:
                dataloader = self.train_loader
                
            # temp solution for lr scheduler
            steps_total = len(self.train_loader)
            steps_acc = self._opt['GRADIENT_ACCUMULATE_STEP']
            steps_update = steps_total // steps_acc
            self._opt["LR_SCHEDULER_PARAMS"]["steps_update_per_epoch"] = steps_update
        return dataloader

    @staticmethod
    def forward_func(trainer, batch):
        loss = trainer.models['default'](batch)
        return loss

    def forward_step(
        self,
        trainer: DefaultTrainer,
        batch,
        grad_acc_batches: List,
        grad_acc_index: int,
        is_distributed: bool,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        loss_info, sample_size_info, extra_info = {}, {}, {}
        batch = move_batch_to_device(batch, self._opt['device'])
        if self._opt['FP16']:
            # in FP16 mode, DeepSpeed casts the model to FP16, so the input needs to be manually casted to FP16
            batch = cast_batch_to_half(batch)

        # --- Conditional Loss Calculation ---
        if self.student_criterion:
            # --- Student Model Training ---
            # 1. Forward pass through student model
            student_outputs = trainer.models['default'](batch) # Student model forward

            # 2. Prepare targets (extract gt masks)
            targets = {}
            gt_masks_list = []
            valid_batch = True
            for item in batch:
                if 'instances' in item and hasattr(item['instances'], 'gt_masks') and item['instances'].gt_masks.tensor.numel() > 0:
                    gt_masks_list.append(item['instances'].gt_masks.tensor.to(self._opt['device']))
                elif 'gt_masks' in item and item['gt_masks'].numel() > 0: # Check alternative key
                     gt_masks_list.append(item['gt_masks'].to(self._opt['device']))
                else:
                    logger.warning(f"Missing or empty ground truth masks in batch item: {item.get('file_name', 'unknown')}. Skipping batch.")
                    valid_batch = False
                    break # Skip this whole batch if any item is invalid
            
            if valid_batch and gt_masks_list:
                 try:
                     # Find max dimensions
                     max_h = max(m.shape[-2] for m in gt_masks_list)
                     max_w = max(m.shape[-1] for m in gt_masks_list)
                     padded_masks_list = []
                     for m in gt_masks_list:
                         # Assuming masks are [num_instances, H, W] or [H, W]
                         # Handle both cases. If [H,W], add batch dim for pad
                         is_single_mask = len(m.shape) == 2
                         if is_single_mask:
                              m = m.unsqueeze(0) # Add batch dim for padding
                              
                         # Pad height and width
                         pad_h = max_h - m.shape[-2]
                         pad_w = max_w - m.shape[-1]
                         # Pad format is (left, right, top, bottom)
                         padded_m = F.pad(m, (0, pad_w, 0, pad_h), value=self.student_criterion.ignore_index)
                         
                         if is_single_mask:
                              padded_m = padded_m.squeeze(0) # Remove added batch dim
                              
                         padded_masks_list.append(padded_m)
                     
                     # Stack the possibly padded masks. This assumes num_instances=1 or masks are [H,W]
                     # If masks are [num_instances, H, W] and num_instances > 1, this stacking needs adjustment
                     # For simple segmentation, assuming [B, H, W] after processing
                     targets['masks'] = torch.stack(padded_masks_list, dim=0)

                 except Exception as e:
                     logger.error(f"Error processing/padding GT masks for student: {e}. Skipping loss calculation.")
                     targets['masks'] = None
            else:
                targets['masks'] = None # No valid masks found in batch

            # 3. Calculate loss using PixelCriterion
            if targets.get('masks') is not None:
                 loss_dict = self.student_criterion(student_outputs, targets)
                 # Ensure loss requires grad if valid
                 for k, v in loss_dict.items():
                      if torch.is_tensor(v) and not v.requires_grad and v.is_floating_point():
                          v.requires_grad_(True)
            else:
                 # Need to return a zero loss tensor on the correct device that requires grad
                 zero_loss = torch.tensor(0.0, device=self._opt['device'], requires_grad=True)
                 loss_dict = {'loss_sem_seg_ce': zero_loss} 
            
            # 4. Prepare outputs for trainer
            loss_info = {k: v.detach().item() for k, v in loss_dict.items()}
            total_loss = sum(loss_dict.values())
            # --- End Student Model Training ---
        else:
            # --- Original SEEM Model Training ---
            loss_dict = trainer.compute_loss(self.forward_func, batch)
            loss_info = {k: v.detach().item() for k, v in loss_dict.items()}
            total_loss = sum(loss for loss in loss_dict.values())
            # --- End Original SEEM Model Training ---

        # Ensure total_loss is a tensor on the correct device before backward pass
        if not torch.is_tensor(total_loss):
            # If loss_dict was empty or resulted in a non-tensor sum (like 0), create a zero tensor.
            logger.warning(f"total_loss was not a tensor ({type(total_loss)}), creating zero tensor for backward pass.")
            total_loss = torch.tensor(0.0, device=trainer.opt['device'], requires_grad=True)
        elif total_loss.device != trainer.opt['device']:
            logger.warning(f"total_loss was on device {total_loss.device}, moving to {trainer.opt['device']}.")
            total_loss = total_loss.to(trainer.opt['device'])
        
        # Check if requires_grad needs to be set (e.g., if created from tensor(0.0))
        if total_loss.is_floating_point() and not total_loss.requires_grad:
             logger.warning("Setting requires_grad=True for total_loss.")
             total_loss.requires_grad_(True)

        sample_size_info = {'num_samples': len(batch)}
        extra_info = {}

        # Backward pass uses the total_loss
        trainer.backward_loss(total_loss, model_names=['default'])
        trainer.update_model(model_name='default')

        return loss_info, sample_size_info, extra_info

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
        save_folder,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        model = trainer.raw_models['default'].eval()
        # Only call hook_opt if ATTENTION_ARCH is defined, as it modifies that section
        if self._opt.get('ATTENTION_ARCH') is not None:
            self._opt = hook_opt(self._opt)
        else:
            logger.info("Skipping hook_opt call as ATTENTION_ARCH is not defined in the config.")
            
        dataset_names = self._opt['DATASETS']['TEST']
        
        scores = {}
        summary = {}

        save_logits_mode = self._opt.get('EVAL', {}).get('SAVE_LOGITS', False)
        base_logits_save_dir = None
        if save_logits_mode:
            base_logits_save_dir = self._opt.get('EVAL', {}).get('LOGITS_SAVE_DIR', 'teacher_logits')
            if not base_logits_save_dir:
                 logger.warning("EVAL.LOGITS_SAVE_DIR is not specified or empty. Using default 'teacher_logits'.")
                 base_logits_save_dir = 'teacher_logits'
            logger.info(f"Logit saving mode enabled. Base directory: {base_logits_save_dir}")

        for dataset_label in dataset_names:
            logger.info(f"Processing dataset: {dataset_label}")
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            
            current_logits_save_dir = None
            if save_logits_mode:
                 clean_dataset_name = re.sub(r'^biomed_', '', dataset_label)
                 clean_dataset_name = re.sub(r'_test$', '', clean_dataset_name)
                 clean_dataset_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', clean_dataset_name)
                 current_logits_save_dir = os.path.join(base_logits_save_dir, clean_dataset_name)
                 os.makedirs(current_logits_save_dir, exist_ok=True)
                 if is_main_process(): 
                     logger.info(f"Target logit directory for {dataset_label}: {current_logits_save_dir}")

            if not save_logits_mode:
                self.evaluator.reset()

            with torch.no_grad():
                names = get_class_names(dataset_label)
                if self._opt['MODEL']['ENCODER']['BINARY_CLASSES']:
                    names = ['target', 'background']
                
                if hasattr(model.model, 'metadata'):
                    model.model.metadata = MetadataCatalog.get(dataset_label)
                    model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                    eval_type = model.model.metadata.evaluator_type
                else:
                    logger.warning("Model does not have metadata attribute. Using default settings.")
                    eval_type = 'default_eval'

                if 'background' in names:
                    if hasattr(model.model, 'sem_seg_head') and model.model.sem_seg_head:
                         model.model.sem_seg_head.num_classes = len(names) - 1
                    else:
                         logger.warning("Model does not have sem_seg_head. Skipping class count adjustment.")

                if hasattr(model.model, 'sem_seg_head') and model.model.sem_seg_head and \
                   hasattr(model.model.sem_seg_head, 'predictor') and model.model.sem_seg_head.predictor and \
                   hasattr(model.model.sem_seg_head.predictor, 'lang_encoder') and model.model.sem_seg_head.predictor.lang_encoder:
                    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                else:
                    logger.warning("Model structure does not allow setting text embeddings. Skipping.")

                hook_switcher(model.model, dataset_label)
                total = len(eval_batch_gen)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                start_data_time = time.perf_counter()

                for idx, batch in enumerate(eval_batch_gen):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0

                    start_compute_time = time.perf_counter()
                    
                    images_tensor = [x["image"].to(model.opt['device']) for x in batch]
                    # Normalization should be handled by the DatasetMapper, remove manual normalization here
                    # images_tensor = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images_tensor]
                    # Get size_divisibility from the configuration instead of the model attribute
                    size_divisibility = self._opt.get('MODEL', {}).get('DECODER', {}).get('SIZE_DIVISIBILITY', 32) # Default to 32 if not found
                    images = ImageList.from_tensors(images_tensor, size_divisibility)

                    if save_logits_mode:
                        features = model.model.backbone(images.tensor)
                        outputs = model.model.sem_seg_head(features)
                    else:
                        batch_device = move_batch_to_device(batch, self._opt['device'])
                        if self._opt['FP16']:
                            batch_device = cast_batch_to_half(batch_device)
                        outputs = model(batch_device, mode=eval_type)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    if save_logits_mode:
                        logits_to_save = None
                        output_data = None
                        if isinstance(outputs, dict):
                             output_data = outputs
                        else:
                             logger.warning(f"Unexpected output type from sem_seg_head: {type(outputs)}")

                        logit_key_found = None
                        possible_logit_keys = ['pred_logits', 'sem_seg_logits']
                        if output_data:
                           for key in possible_logit_keys:
                               if key in output_data:
                                   logits_to_save = output_data[key].detach().cpu()
                                   logit_key_found = key
                                   break

                        if logits_to_save is not None:
                            try:
                                file_name = os.path.basename(batch[0]['file_name'])
                                save_name = file_name.split('.')[0] + ".pt"
                            except (KeyError, IndexError, TypeError):
                                logger.warning(f"Could not get file_name from batch {idx} for {dataset_label}. Using index.")
                                save_name = f"output_{idx}.pt"

                            save_path = os.path.join(current_logits_save_dir, save_name)
                            if idx == 0 or idx % 200 == 0:
                                logger.info(f"[{dataset_label}] Saving logits (key: '{logit_key_found}') to {save_path}")
                            torch.save(logits_to_save, save_path)
                        else:
                            if idx < 5 or idx % 100 == 0:
                                available_keys = list(output_data.keys()) if output_data else []
                                logger.warning(f"Could not find expected logit keys {possible_logit_keys} in model head output for batch {idx} (Dataset: {dataset_label}). Available keys: {available_keys}")
                    else:
                        start_eval_time = time.perf_counter()
                        self.evaluator.process(batch_device, outputs)
                        total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    
                    log_message_base = (
                        f"Task {dataset_label}. "
                        f"Processing {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                    )
                    if not save_logits_mode:
                         eval_seconds_per_iter = total_eval_time / iters_after_start
                         total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                         eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                         log_message = log_message_base + (
                             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                             f"ETA={eta}"
                         )
                    else:
                        log_message = log_message_base + "(Saving logits)"

                    if is_main_process() and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5 or save_logits_mode):
                         log_every_n_seconds(
                             logging.INFO,
                             log_message,
                             n=5,
                         )

                    start_data_time = time.perf_counter()
            
            if not save_logits_mode:
                results = self.evaluator.evaluate()
                if hasattr(model.model, 'sem_seg_head') and model.model.sem_seg_head and \
                   hasattr(model.model.sem_seg_head, 'predictor') and model.model.sem_seg_head.predictor and \
                   hasattr(model.model.sem_seg_head.predictor, 'lang_encoder') and model.model.sem_seg_head.predictor.lang_encoder:
                    model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()

                if is_main_process():
                    scores["{}/{}".format(dataset_label, eval_type)] = results
            else:
                if is_main_process():
                    logger.info(f"Finished saving logits for dataset: {dataset_label}")

        if is_main_process() and not save_logits_mode and scores:
            model_name_part = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            save_path = os.path.join(save_folder, f'{model_name_part}_eval_results.json')
            logger.info(f"Saving evaluation results to {save_path}")
            with open(save_path, 'w') as f:
                json.dump(scores, f, indent=4)

            processed_scores = {}
            for datatype in scores:
                processed_scores[datatype] = {}
                for evaltype in scores[datatype]:
                    if isinstance(scores[datatype][evaltype], dict) and 'instance_results' in scores[datatype][evaltype]:
                         processed_scores[datatype][evaltype] = scores[datatype][evaltype]['scores']
                    elif isinstance(scores[datatype][evaltype], dict):
                         processed_scores[datatype][evaltype] = scores[datatype][evaltype]
                    else:
                         processed_scores[datatype][evaltype] = scores[datatype][evaltype]
            return processed_scores
        elif save_logits_mode:
             logger.info("Logit saving complete for all specified datasets.")
             return {}
        else:
             logger.info("No scores generated or saved.")
             return {}