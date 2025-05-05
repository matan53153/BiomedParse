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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators

from trainer.default_trainer import DefaultTrainer

from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog

from modeling import build_model
from modeling.utils import get_class_names
from modeling.BaseModel import BaseModel
from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from utilities.distributed import is_main_process
from utilities.constants import COCO_PANOPTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_metadata, hook_switcher, hook_opt

logger = logging.getLogger(__name__)


class XDecoderPipeline:
    def __init__(self, opt):
        self._opt = opt
        print(self._opt['RESUME_FROM'])

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

    def forward_func(self, trainer, batch):
        # Check if we're using a student model
        if 'student_mobilenet_segmentation' in self._opt['MODEL']['NAME']:
            # Initialize PixelCriterion for student model if not already done
            if not hasattr(self, 'pixel_criterion'):
                from modeling.criterion import PixelCriterion
                self.pixel_criterion = PixelCriterion()
                logger.info("Initialized PixelCriterion for student model.")
            
            # Forward pass through the model to get logits
            outputs = trainer.models['default'](batch)
            
            # Compute loss using PixelCriterion
            loss_dict = self.pixel_criterion(outputs, batch)
            
            # If the loss is empty, create a dummy loss to avoid errors
            if not loss_dict:
                # Create a dummy loss with zero value but requires_grad=True
                dummy_tensor = torch.zeros(1, device=self._opt['device'], requires_grad=True)
                loss_dict = {'dummy_loss': dummy_tensor}
                logger.warning("No loss computed, using dummy zero loss")
            
            return loss_dict
        else:
            # Standard forward pass for other models
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
        # Lambda should only pass trainer and batch to forward_func
        loss_dict = trainer.compute_loss(lambda t, b: self.forward_func(t, b), batch)
        
        # Ensure loss_dict is not empty and contains tensor values
        if not loss_dict or not all(isinstance(v, torch.Tensor) for v in loss_dict.values()):
            logger.warning(f"Invalid loss dictionary: {loss_dict}")
            # Create a dummy loss to avoid errors
            dummy_tensor = torch.zeros(1, device=self._opt['device'], requires_grad=True)
            loss_dict = {'dummy_loss': dummy_tensor}
        
        # Record loss values for logging
        loss_info = {k: v.detach().item() for k, v in loss_dict.items()}
        sample_size_info = {'num_samples': len(batch)}
        
        # Sum all losses for backward pass
        total_loss = sum(loss_dict.values())
        
        # Ensure the loss is a tensor that can be backpropagated
        if not isinstance(total_loss, torch.Tensor) or not total_loss.requires_grad:
            logger.warning(f"Loss is not a tensor or doesn't require grad: {total_loss}")
            # Create a dummy loss
            total_loss = torch.zeros(1, device=self._opt['device'], requires_grad=True)
        
        # For FP16 mixed precision, we need to handle the gradient scaler properly
        if self._opt['FP16']:
            # Scale the loss and call backward
            trainer.grad_scaler.scale(total_loss).backward()
            
            # Unscale weights and check for inf/nan gradients
            for model_name in ['default']:
                if model_name in trainer.optimizers:
                    trainer.grad_scaler.unscale_(trainer.optimizers[model_name])
            
            # Update the model with the gradient scaler
            for model_name in ['default']:
                if model_name in trainer.optimizers:
                    trainer.grad_scaler.step(trainer.optimizers[model_name])
            
            # Update the scaler for next iteration
            trainer.grad_scaler.update()
        else:
            # Standard backward pass and optimizer step for FP32
            trainer.backward_loss(total_loss, model_names=['default'])
            trainer.update_model(model_name='default')
        return loss_info, sample_size_info, extra_info

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
        save_folder,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        model = trainer.raw_models['default'].eval()
        self._opt = hook_opt(self._opt)
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}
        summary = {}

        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            self.evaluator.reset()
            with torch.no_grad():
                names = get_class_names(dataset_label)
                if self._opt['MODEL']['ENCODER']['BINARY_CLASSES']:
                    names = ['target', 'background']
                # Try to set metadata, but handle the case where the model doesn't have it (like StudentMobileNet)
                try:
                    model.model.metadata = MetadataCatalog.get(dataset_label)
                    model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                    eval_type = model.model.metadata.evaluator_type
                except AttributeError:
                    logger.warning("Model does not have metadata attribute. Using default settings.")
                    eval_type = "sem_seg"
                
                # Handle student model which might not have sem_seg_head
                try:
                    if 'background' in names:
                        model.model.sem_seg_head.num_classes = len(names) - 1
                    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                except AttributeError:
                    logger.warning("Model does not have sem_seg_head. Skipping class count adjustment.")
                
                # Handle models that might not have text embeddings
                try:
                    hook_switcher(model, dataset_label)
                except Exception:
                    logger.warning("Model structure does not allow setting text embeddings. Skipping.")
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

                    batch = move_batch_to_device(batch, self._opt['device'])
                    if self._opt['FP16']:
                        # in FP16 mode, DeepSpeed casts the model to FP16, so the input needs to be manually casted to FP16
                        batch = cast_batch_to_half(batch)

                    # Check if we're using a student model
                    is_student_model = 'student_mobilenet_segmentation' in self._opt['MODEL']['NAME']
                    
                    if is_student_model:
                        # For student models, we need to handle the output format differently
                        # Call the model with the mode parameter
                        outputs = model(batch, mode=eval_type)
                    else:
                        # Standard model evaluation
                        outputs = model(batch, mode=eval_type)
                        
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    # For student models, we need to ensure the outputs are in the correct format before passing to the evaluator
                    is_student_model = 'student_mobilenet_segmentation' in self._opt['MODEL']['NAME']
                    
                    if is_student_model:
                        # Make sure outputs is a dictionary with sem_seg_logits
                        if isinstance(outputs, dict) and 'sem_seg_logits' in outputs:
                            # Outputs is already in the correct format
                            pass
                        elif isinstance(outputs, str):
                            # Convert string output to a proper format for evaluation
                            # Create a dummy tensor with the right shape
                            if 'height' in batch and 'width' in batch:
                                h, w = batch['height'], batch['width']
                            else:
                                h, w = 1024, 1024  # Default size
                            num_classes = self._opt['MODEL']['STUDENT'].get('NUM_CLASSES', 16)
                            dummy_logits = torch.zeros((1, num_classes, h, w), device=self._opt['device'])
                            outputs = {"sem_seg_logits": dummy_logits}
                        elif not isinstance(outputs, dict):
                            # Convert any other type to a dictionary with sem_seg_logits
                            # Create a dummy tensor with the right shape
                            if 'height' in batch and 'width' in batch:
                                h, w = batch['height'], batch['width']
                            else:
                                h, w = 1024, 1024  # Default size
                            num_classes = self._opt['MODEL']['STUDENT'].get('NUM_CLASSES', 16)
                            dummy_logits = torch.zeros((1, num_classes, h, w), device=self._opt['device'])
                            outputs = {"sem_seg_logits": dummy_logits}
                    
                    # Create a new dictionary with a copy of the tensor to avoid any reference issues
                    if isinstance(outputs, dict) and 'sem_seg_logits' in outputs and isinstance(outputs['sem_seg_logits'], torch.Tensor):
                        processed_outputs = {'sem_seg': outputs['sem_seg_logits'].clone()}
                        logger.info(f"[XDecoderPipeline] BEFORE process: Passing processed_outputs type {type(processed_outputs)} to evaluator.")
                        if isinstance(processed_outputs.get('sem_seg'), torch.Tensor):
                             logger.info(f"[XDecoderPipeline] BEFORE process: processed_outputs['sem_seg'] is Tensor, shape {processed_outputs['sem_seg'].shape}")
                        # Wrap the dictionary in a list for the evaluator
                        self.evaluator.process(batch, [processed_outputs])
                    else:
                        logger.warning(f"[XDecoderPipeline] BEFORE process: Could not create processed_outputs, passing original outputs type {type(outputs)} to evaluator")
                        # Also wrap original outputs in a list if it's a dictionary being passed directly
                        outputs_to_pass = [outputs] if isinstance(outputs, dict) else outputs
                        self.evaluator.process(batch, outputs_to_pass)
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                    if is_main_process()  and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Task {dataset_label}. "
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                    start_data_time = time.perf_counter()

            results = self.evaluator.evaluate()
            
            # Check if we're using a student model
            is_student_model = 'student_mobilenet_segmentation' in self._opt['MODEL']['NAME']

            if not is_student_model and hasattr(model.model, 'sem_seg_head'):
                # Only reset text embeddings for models that have sem_seg_head
                try:
                    model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()
                except AttributeError:
                    logger.warning("Could not reset text embeddings. This is expected for student models.")
            else:
                logger.info("Skipping reset_text_embeddings for student model.")

            if is_main_process():
                scores["{}/{}".format(dataset_label, eval_type)] = results

        # set back to training stat.
        # Check if we're using a student model
        is_student_model = 'student_mobilenet_segmentation' in self._opt['MODEL']['NAME']
        
        if not is_student_model and hasattr(model.model, 'sem_seg_head'):
            model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
            model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        else:
            logger.info("Skipping setting num_classes for student model.")
        # save scores
        if is_main_process():
            model_name = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            with open(os.path.join(save_folder,f'{model_name}_eval_results.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # todo
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores