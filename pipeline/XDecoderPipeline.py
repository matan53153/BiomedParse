# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import datetime
import logging
import time
import json
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators
from omegaconf import OmegaConf

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

        # --- Load Teacher Model for Knowledge Distillation ---
        self.teacher_model = None
        self.kd_enabled = self._opt.get('KD', {}).get('ENABLED', False)
        if self.kd_enabled:
            logger.info("Knowledge Distillation enabled. Loading teacher model...")
            kd_opt = self._opt['KD']
            self.kd_alpha = kd_opt['ALPHA']
            self.kd_temperature = kd_opt['TEMPERATURE']

            # Load teacher config
            teacher_config_path = os.path.join(self._opt['base_path'], kd_opt['TEACHER_CONFIG_PATH'])
            teacher_opt = OmegaConf.load(teacher_config_path)

            # Merge necessary options from student config (like device, base_path)
            # Be careful not to overwrite essential teacher model parameters
            teacher_opt_merged = copy.deepcopy(teacher_opt)
            OmegaConf.set_struct(teacher_opt_merged, False) # Make it possible to add new keys like KD_TEACHER_MODE
            teacher_opt_merged.base_path = self._opt['base_path']
            teacher_opt_merged.FP16 = self._opt['FP16'] # Ensure consistency
            OmegaConf.set_struct(teacher_opt_merged, True) # Make it struct again

            # Build teacher model
            teacher_model_raw = build_model(teacher_opt_merged)

            # Load teacher weights
            teacher_weights_path = kd_opt['TEACHER_WEIGHTS_PATH']
            if not os.path.isabs(teacher_weights_path):
                 teacher_weights_path = os.path.join(self._opt['base_path'], teacher_weights_path)

            if os.path.exists(teacher_weights_path):
                logger.info(f"Loading teacher weights from: {teacher_weights_path}")
                # Map location based on current device
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self._opt['rank']} if self._opt['rank'] != -1 else self._opt['device']
                checkpoint = torch.load(teacher_weights_path, map_location=map_location)
                # Adjust for potential model wrapping (e.g., DDP)
                state_dict = checkpoint.get('model', checkpoint) 
                # Handle potential DataParallel prefix 'module.'
                if next(iter(state_dict)).startswith('module.'):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                
                incompatible_keys = teacher_model_raw.load_state_dict(state_dict, strict=False)
                if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                     logger.warning(f"Teacher weight loading mismatch: {incompatible_keys}")
                else:
                     logger.info("Teacher weights loaded successfully.")

            else:
                logger.warning(f"Teacher weights file not found at {teacher_weights_path}. KD might not work correctly without pre-trained teacher.")

            # Prepare teacher model for inference
            self.teacher_model = teacher_model_raw.to(self._opt['device'])
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            logger.info("Teacher model loaded and frozen.")

            # Explicitly generate text embeddings for the teacher model
            try:
                # Get class names from the teacher config (assuming standard structure)
                # Ensure the dataset name extraction is correct for your config format
                teacher_dataset_name = teacher_opt_merged.DATASETS.TRAIN[0] 
                teacher_class_names = get_class_names(teacher_dataset_name)
                
                logger.info(f"Generating text embeddings for teacher model (dataset: {teacher_dataset_name})...")
                # Access the language encoder via sem_seg_head.predictor
                self.teacher_model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    teacher_class_names, 
                    name='default', # Name used in compute_similarity
                    is_eval=True    # Use evaluation mode templates/logic
                )
                logger.info("Teacher text embeddings generated successfully.")
            except Exception as e:
                logger.error(f"Critical error: Failed to generate teacher text embeddings: {e}")
                logger.error("Knowledge Distillation cannot proceed without teacher embeddings.")
                # Depending on requirements, you might want to disable KD or raise the error
                raise RuntimeError("Failed to initialize teacher embeddings for KD") from e

        else:
            logger.info("Knowledge Distillation is disabled.")

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
            if not hasattr(self, 'train_loader'):G
                dataloader = build_train_dataloader(seGf._opt)
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

    def forward_step(self, trainer, batch):
        """Perform a single training step, including KD if enabled."""
        # Move batch to the correct device
        batch = move_batch_to_device(batch, self._opt['device'])
        if self._opt['FP16']:
            batch = cast_batch_to_half(batch)

        if self.kd_enabled and self.teacher_model is not None:
            # --- Knowledge Distillation Step --- 

            # 1. Get task loss from student model (training mode)
            # Assumes model returns dict with e.g., {'loss_sem_seg_ce': tensor(...)}
            student_loss_dict = trainer.models['default'](batch)
            task_loss = student_loss_dict.get('loss_sem_seg_ce', torch.tensor(0.0, device=self._opt['device']))

            # 2. Get student logits (inference mode)
            # Assumes model returns dict with e.g., {'sem_seg': tensor(...)}
            student_outputs = trainer.models['default'](batch)
            student_logits = student_outputs.get("sem_seg") 

            # 3. Get teacher logits (inference mode, no grad)
            with torch.no_grad():
                # Assumes teacher model returns dict like student in inference
                teacher_outputs = self.teacher_model(batch)
                teacher_logits = teacher_outputs.get("sem_seg") 

            loss_info = student_loss_dict.copy() # Start with task loss info

            # 4. Calculate KD loss (if logits are available)
            kd_loss = torch.tensor(0.0, device=self._opt['device'])
            if student_logits is not None and teacher_logits is not None:
                # Ensure logits are same spatial size (simple upsample if needed)
                if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
                    student_logits = F.interpolate(student_logits, size=teacher_logits.shape[-2:], mode='bilinear', align_corners=False)

                # Soften probabilities with temperature
                student_log_probs = F.log_softmax(student_logits / self.kd_temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=1)

                # KL Divergence Loss (batchmean averages over N, H, W)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean', log_target=False) * (self.kd_temperature ** 2)
                
                loss_info['loss_kd'] = kd_loss.item() # Log KD loss value
            else:
                logger.warning("KD Loss calculation skipped: student or teacher logits were None.")
                loss_info['loss_kd'] = 0.0 # Log 0 if KD couldn't be calculated

            # 5. Combine losses
            total_loss = (1.0 - self.kd_alpha) * task_loss + self.kd_alpha * kd_loss
            
            # Update the main loss key used for backprop
            # Ensure the key matches what the trainer expects (usually 'loss')
            loss_info['loss'] = total_loss 
            # Add total task loss for logging if needed (before weighting)
            loss_info['task_loss'] = task_loss.item() 

            return loss_info # Return combined loss dict

        else:
            # --- Original Training Step (No KD) --- 
            # Base forward function to pass to compute_loss 
            def base_forward_func(t, b):
                # Assumes model's forward(is_training=True) returns loss dict
                return t.models['default'](b)
            
            # compute_loss handles the forward pass and loss calculation internally
            loss_dict = trainer.compute_loss(base_forward_func, batch)
            return loss_dict

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
                model.model.metadata = MetadataCatalog.get(dataset_label)
                model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                eval_type = model.model.metadata.evaluator_type
                if 'background' in names:
                    model.model.sem_seg_head.num_classes = len(names) - 1
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                hook_switcher(model, dataset_label)
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
                        batch = cast_batch_to_half(batch)

                    outputs = model(batch, mode=eval_type)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    self.evaluator.process(batch, outputs)
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
            model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()

            if is_main_process():
                scores["{}/{}".format(dataset_label, eval_type)] = results

        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        # save scores
        if is_main_process():
            model_name = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(save_folder,f'{model_name}_eval_results_{timestamp}.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
        if is_main_process():
            model_name = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(save_folder,f'{model_name}_eval_results_{timestamp}.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(save_folder,f'{model_name}_eval_results_{timestamp}.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
        return scores
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores

            results = self.evaluator.evaluate()
            model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()

            if is_main_process():
                scores["{}/{}".format(dataset_label, eval_type)] = results

        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        # save scores
        if is_main_process():
            model_name = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(save_folder,f'{model_name}_eval_results_{timestamp}.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores
            if is_main_process():
                scores["{}/{}".format(dataset_label, eval_type)] = results

        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        # save scores
        if is_main_process():
            model_name = self._opt['RESUME_FROM'].split('/')[-1].split('.')[0]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(save_folder,f'{model_name}_eval_results_{timestamp}.json'), 'w') as f:
                json.dump(scores, f, indent=4)
        # hack to return only results/scores 
        for datatype in scores:
            for evaltype in scores[datatype]:
                if 'instance_results' in scores[datatype][evaltype]:
                    scores[datatype][evaltype]= scores[datatype][evaltype]['scores']
        return scores