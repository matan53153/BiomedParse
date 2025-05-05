# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import logging
import torch
from torchvision.ops import box_iou

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator

import matplotlib.pyplot as plt
import numpy as np
import os

import copy

class GroundingEvaluator(DatasetEvaluator):
    """
    Evaluate grounding segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        output_dir,
        compute_box=False,
        distributed=True,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._compute_box = compute_box
        meta = MetadataCatalog.get(dataset_name)

    def reset(self):
        self.cum_I = 0
        self.cum_U = 0
        self.mIoU = 0
        self.mDice = 0
        self.cum_mean_area = 0
        self.eval_seg_iou_list = [.5, .6, .7, .8, .9]
        self.seg_correct = torch.zeros(len(self.eval_seg_iou_list), device=self._cpu_device)
        self.seg_total = 0
        self.instance_results = []
        if self._compute_box:
            self.mIoU_box = 0
            self.seg_correct_box = torch.zeros(len(self.eval_seg_iou_list), device=self._cpu_device)

    @staticmethod
    def computeIoU(pred_seg, gd_seg):
        I = (pred_seg & gd_seg)
        U = (pred_seg | gd_seg)
        return I, U

    def get_metadata(self, _input):
        """
        Extracts and returns specific metadata from the input dictionary.
        
        Parameters:
        _input (dict): A dictionary containing keys like 'file_name', 'image_id', and 'grounding_info'.
                    The 'grounding_info' is a list of dictionaries with keys like 'area', 'iscrowd', etc.
        
        Returns:
        dict: A dictionary containing filtered metadata.
        """

        _input = copy.deepcopy(_input)

        selected_input_keys = ['file_name', 'image_id', 'grounding_info']
        selected_grounding_info_keys = ['area', 'mask_file', 'iscrowd', 'image_id', 'category_id', 'id', 'file_name', 'split', 'ann_id', 'ref_id']

        filtered_input = {key: _input[key] for key in selected_input_keys if key in _input}

        # Check if grounding_info is present and is a list
        if 'grounding_info' in filtered_input and isinstance(filtered_input['grounding_info'], list):
            # Filter each grounding_info dictionary
            filtered_input['grounding_info'] = [
                {key: info[key] for key in selected_grounding_info_keys if key in info}
                for info in filtered_input['grounding_info']
            ]

        return filtered_input

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # Get predicted mask (binary threshold at 0.5)
            pred = output['grounding_mask'].sigmoid() > 0.5
            
            # Get ground truth mask
            gt = input['groundings']['masks'].bool()
            
            # Save both ground truth and predicted masks as images
            # Create directory for saving segmentation results
            save_dir = os.path.join(os.getcwd(), 'segmentation_results')
            os.makedirs(save_dir, exist_ok=True)
            
            # Get image identifier from file name
            if 'file_name' in input:
                img_id = os.path.basename(input['file_name']).split('.')[0]
            else:
                img_id = f"image_{input.get('image_id', 'unknown')}"
                
            # Get text prompt if available
            if 'groundings' in input and 'texts' in input['groundings'] and len(input['groundings']['texts']) > 0:
                text_prompt = input['groundings']['texts'][0].replace(' ', '+')
            else:
                text_prompt = 'unknown_prompt'
            
            # Create base filename
            base_filename = f"{img_id}_{text_prompt}"
            
            # Save probability map (raw sigmoid output)
            prob = output['grounding_mask'].sigmoid().cpu().numpy()[0] * 255
            prob_file = os.path.join(save_dir, f"{base_filename}_prob.png")
            plt.imsave(prob_file, prob.astype(np.uint8), cmap='gray')
            
            # Save predicted binary mask
            pred_mask = pred[0].cpu().numpy().astype(np.uint8) * 255
            pred_file = os.path.join(save_dir, f"{base_filename}_pred.png")
            plt.imsave(pred_file, pred_mask, cmap='gray')
            
            # Save ground truth mask
            gt_mask = gt[0].cpu().numpy().astype(np.uint8) * 255
            gt_file = os.path.join(save_dir, f"{base_filename}_gt.png")
            plt.imsave(gt_file, gt_mask, cmap='gray')
            
            # Save overlay visualization (ground truth in red, prediction in green, overlap in yellow)
            overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            overlay[..., 0] = gt_mask  # Red channel for ground truth
            overlay[..., 1] = pred_mask  # Green channel for prediction
            overlay_file = os.path.join(save_dir, f"{base_filename}_overlay.png")
            plt.imsave(overlay_file, overlay)

            gt = input['groundings']['masks'].bool()
            bsi = len(pred)
            I, U = self.computeIoU(pred, gt)
            self.cum_I += I.sum().cpu()
            self.cum_U += U.sum().cpu()
            IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
            self.mIoU += IoU.sum().cpu()
            # Add Dice score in eval
            Dice = I.reshape(bsi,-1).sum(-1)*2.0 / (gt.reshape(bsi,-1).sum(-1) + pred.reshape(bsi,-1).sum(-1) + 1e-6)
            self.mDice += Dice.sum().cpu()
            self.cum_mean_area += ((gt.reshape(bsi,-1).sum(-1) + pred.reshape(bsi,-1).sum(-1)) / 2.0).sum().cpu()

            if self._compute_box:
                pred_box = BoxMode.convert(output['grounding_box'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                gt_box = BoxMode.convert(input['groundings']['boxes'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS).cpu()
                IoU_box = box_iou(pred_box, gt_box).diagonal()
                self.mIoU_box += IoU_box.sum()

            for idx in range(len(self.eval_seg_iou_list)):
                eval_seg_iou = self.eval_seg_iou_list[idx]
                self.seg_correct[idx] += (IoU >= eval_seg_iou).sum().cpu()
                if self._compute_box:
                    self.seg_correct_box[idx] += (IoU_box >= eval_seg_iou).sum().cpu()
            self.seg_total += bsi

            instance_result = {
                'metadata': self.get_metadata(input),
                'IoU': IoU.cpu().numpy().tolist(),      
                'Dice': Dice.cpu().numpy().tolist(),
                'I': I.sum(dim=(1, 2)).cpu().numpy().tolist(),
                'U': U.sum(dim=(1, 2)).cpu().numpy().tolist(),
                'IoU_box': IoU_box.cpu().numpy().tolist() if self._compute_box else '',
                'pred_area': pred.reshape(bsi,-1).sum(-1).cpu().numpy().tolist(),
            }

            iou_len = IoU.shape[0]
            grounding_info_len = len(self.get_metadata(input)['grounding_info'])
            assert iou_len == grounding_info_len, f'Number of IoU scores ({iou_len}) and grounding info ({grounding_info_len}) do not match.'   
            self.instance_results.append(instance_result)

    def evaluate(self):
        if self._distributed:
            synchronize()
            self.cum_I = torch.stack(all_gather(self.cum_I)).sum()
            self.cum_U = torch.stack(all_gather(self.cum_U)).sum()
            self.mIoU = torch.stack(all_gather(self.mIoU)).sum()
            self.mDice = torch.stack(all_gather(self.mDice)).sum()
            self.cum_mean_area = torch.stack(all_gather(self.cum_mean_area)).sum()
            self.seg_correct = torch.stack(all_gather(self.seg_correct)).sum(0)
            self.seg_total = sum(all_gather(self.seg_total))
            self.instance_results = sum(all_gather(self.instance_results), [])
            if self._compute_box:
                self.mIoU_box = torch.stack(all_gather(self.mIoU_box)).sum()
                self.seg_correct_box = torch.stack(all_gather(self.seg_correct_box)).sum(0)
            if not is_main_process():
                return

        results = {}
        for idx in range(len(self.eval_seg_iou_list)):
            result_str = 'precision@{}'.format(self.eval_seg_iou_list[idx])
            results[result_str] = (self.seg_correct[idx]*100 / self.seg_total).item()
        results['cIoU'] = (self.cum_I*100./self.cum_U).item()
        results['mIoU'] = (self.mIoU*100./self.seg_total).item()
        results['cDice'] = (self.cum_I*100./self.cum_mean_area).item()
        results['mDice'] = (self.mDice*100./self.seg_total).item()

        if self._compute_box:
            for idx in range(len(self.eval_seg_iou_list)):
                result_str = 'precisionB@{}'.format(self.eval_seg_iou_list[idx])
                results[result_str] = (self.seg_correct_box[idx]*100 / self.seg_total).item()
            results['mBIoU'] = (self.mIoU_box*100./self.seg_total).item()

        self._logger.info(results)
        
        # Save detailed evaluation results to a JSON file
        import json
        import os
        from datetime import datetime
        
        # Extract model/run name from the output_dir path
        model_run_name = os.path.basename(self._output_dir.rstrip('/\\')) 
        
        # Create results directory structure: base_results_dir/model_name/
        base_results_dir = os.path.join(os.getcwd(), 'evaluation_results') 
        model_results_dir = os.path.join(base_results_dir, model_run_name)
        os.makedirs(model_results_dir, exist_ok=True) 

        # Create a timestamped filename within the model-specific folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(model_results_dir, f'{self._dataset_name}_eval_{timestamp}.json')
        
        # Save results and instance-level data
        with open(results_file, 'w') as f:
            json.dump({
                'model_run': model_run_name, 
                'dataset': self._dataset_name,
                'metrics': results,
                'instance_results': self.instance_results
            }, f, indent=2)
        
        self._logger.info(f'Saved detailed evaluation results to {results_file}')
        
        return {'grounding': {'scores': results, 'instance_results': self.instance_results}}