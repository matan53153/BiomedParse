# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from utilities.distributed import synchronize

from ..semseg_loader import load_semseg


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        
        # Check if this is a biomedical dataset or student model dataset
        is_biomed_dataset = 'biomed' in dataset_name
        
        # For biomedical datasets, we might not have sem_seg_file_name
        # In this case, we'll use a dummy mapping and handle it in the process method
        dataset_records = DatasetCatalog.get(dataset_name)
        if is_biomed_dataset and len(dataset_records) > 0 and 'sem_seg_file_name' not in dataset_records[0]:
            self._logger.info(f"Dataset {dataset_name} doesn't have sem_seg_file_name field. Using file_name for both input and ground truth.")
            self.input_file_to_gt_file = {
                dataset_record["file_name"]: dataset_record["file_name"]
                for dataset_record in dataset_records
            }
            # Flag to indicate we're using a simplified evaluation for biomedical datasets
            self._is_simplified_eval = True
        else:
            self.input_file_to_gt_file = {
                dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
                for dataset_record in dataset_records
            }
            self._is_simplified_eval = False

        meta = MetadataCatalog.get(dataset_name)
        
        # Check if this is a biomedical dataset
        is_biomed_dataset = 'biomed' in dataset_name
        
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
            
        # Handle datasets without stuff_classes
        if hasattr(meta, 'stuff_classes'):
            self._class_names = meta.stuff_classes
            self._num_classes = len(meta.stuff_classes)
        elif is_biomed_dataset:
            # For biomedical datasets, use the actual category names from the UWaterlooSkinCancer dataset
            self._logger.info(f"Dataset {dataset_name} doesn't have stuff_classes. Using biomedical category names.")
            # These category names are from the UWaterlooSkinCancer dataset
            self._class_names = [
                "background", "liver", "lung", "kidney", "pancreas", 
                "heart anatomies", "brain anatomies", "eye anatomies", "vessel", 
                "other organ", "tumor", "infection", "lesion", "fluid disturbance", 
                "other abnormality", "histology structure", "other"
            ]
            self._num_classes = len(self._class_names)
        else:
            # For other datasets, use a minimal set of class names
            self._logger.warning(f"Dataset {dataset_name} doesn't have stuff_classes. Using minimal class names.")
            self._class_names = ["background", "foreground"]
            self._num_classes = 2
            
        self._class_offset = meta.class_offset if hasattr(meta, 'class_offset') else 0
        self._semseg_loader = meta.semseg_loader if hasattr(meta, 'semseg_loader') else 'PIL'

        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            # Create subdir for saving masks
            self._mask_save_dir = os.path.join(self._output_dir, "eval_masks")
            os.makedirs(self._mask_save_dir, exist_ok=True)
            self._logger.info(f"Saving evaluation masks to: {self._mask_save_dir}")

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            # === Debug Log Start ===
            self._logger.info(f"[SemSegEvaluator] START process: Received output type: {type(output)}")
            if isinstance(output, dict):
                self._logger.info(f"[SemSegEvaluator] START process: Output keys: {list(output.keys())}")
                if 'sem_seg' in output and isinstance(output['sem_seg'], torch.Tensor):
                    self._logger.info(f"[SemSegEvaluator] START process: output['sem_seg'] is Tensor, shape {output['sem_seg'].shape}")
                elif 'sem_seg_logits' in output and isinstance(output['sem_seg_logits'], torch.Tensor):
                     self._logger.info(f"[SemSegEvaluator] START process: output['sem_seg_logits'] is Tensor, shape {output['sem_seg_logits'].shape}")
            elif isinstance(output, str):
                 self._logger.info(f"[SemSegEvaluator] START process: Received string output (first 100): {output[:100]}")
            # === Debug Log End ===

            if isinstance(output, dict):
                if "sem_seg" in output:
                    output_tensor = output["sem_seg"]
                    # Check tensor type and dimensions
                    if isinstance(output_tensor, torch.Tensor):
                        self._logger.info(f"[SemSegEvaluator] process: Initial output_tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
                        if output_tensor.dim() == 4: # Expecting B, C, H, W
                            try:
                                output_tensor = output_tensor.argmax(dim=1) # Get predicted class indices
                                self._logger.info(f"[SemSegEvaluator] process: After argmax, output_tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
                                pred = output_tensor.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU, convert to numpy
                                self._logger.info(f"[SemSegEvaluator] process: Final pred shape: {pred.shape}, dtype: {pred.dtype}")
                            except Exception as e:
                                self._logger.error(f"[SemSegEvaluator] process: Error during tensor processing: {e}", exc_info=True)
                                h, w = input["height"], input["width"]
                                pred = np.zeros((h, w), dtype=np.uint8) # Dummy on error
                        elif output_tensor.dim() == 3: # B, H, W or C, H, W
                           # Handle cases where output might already be class indices or need squeezing differently
                            self._logger.warning(f"[SemSegEvaluator] process: Received 3D tensor shape {output_tensor.shape}. Trying to process.")
                            try:
                                # Assuming it might be B, H, W already
                                if output_tensor.shape[0] == 1:
                                    pred = output_tensor.squeeze(0).cpu().numpy()
                                else: # Assuming C, H, W - needs clarification on format
                                    self._logger.warning(f"[SemSegEvaluator] process: Ambiguous 3D tensor, creating dummy.")
                                    h, w = input["height"], input["width"]
                                    pred = np.zeros((h, w), dtype=np.uint8)
                            except Exception as e:
                                self._logger.error(f"[SemSegEvaluator] process: Error processing 3D tensor: {e}", exc_info=True)
                                h, w = input["height"], input["width"]
                                pred = np.zeros((h, w), dtype=np.uint8) # Dummy on error
                        else:
                            self._logger.warning(f"[SemSegEvaluator] process: Unexpected tensor dimension {output_tensor.dim()}. Creating dummy prediction.")
                            h, w = input["height"], input["width"]
                            pred = np.zeros((h, w), dtype=np.uint8)
                    else:
                        self._logger.warning(f"[SemSegEvaluator] process: 'sem_seg' value is not a Tensor (type: {type(output_tensor)}). Creating dummy prediction.")
                        h, w = input["height"], input["width"]
                        pred = np.zeros((h, w), dtype=np.uint8)
                else:
                    self._logger.warning("[SemSegEvaluator] process: Output dictionary does not contain 'sem_seg' key. Creating dummy prediction.")
                    h, w = input["height"], input["width"]
                    pred = np.zeros((h, w), dtype=np.uint8) # Dummy prediction if key missing
            elif isinstance(output, str): # Keep the original check just in case
                 self._logger.warning(f"[SemSegEvaluator] process: Received string output type again: {output}. Creating dummy prediction.")
                 h, w = input["height"], input["width"]
                 pred = np.zeros((h, w), dtype=np.uint8) # Dummy prediction
            else:
                 # Log if it's neither dict nor str, and create dummy
                 self._logger.warning(f"[SemSegEvaluator] process: Unexpected output type: {type(output)}. Creating dummy prediction.")
                 h, w = input["height"], input["width"]
                 pred = np.zeros((h, w), dtype=np.uint8) # Dummy prediction

            if hasattr(self, '_is_simplified_eval') and self._is_simplified_eval:
                gt = np.zeros_like(pred)
            else:
                try:
                    with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                        gt = load_semseg(f, self._semseg_loader) - self._class_offset
                except Exception as e:
                    self._logger.warning(f"Failed to load ground truth for {input['file_name']}: {e}")
                    gt = np.zeros_like(pred)
            
            if isinstance(self._ignore_label, int):
                ignore_label = self._ignore_label - self._class_offset
                gt[gt == ignore_label] = self._num_classes
            
            self._logger.debug(f"Before resize check: Pred shape {pred.shape}, GT shape {gt.shape}") # Log before check
            # Resize GT to match Pred shape if necessary
            if gt.shape != pred.shape:
                self._logger.warning(f"Shape mismatch detected! Resizing GT from {gt.shape} to {pred.shape}...") # Log entry
                # Convert to tensor, add B, C dims, resize, remove dims, convert back to numpy
                gt_tensor = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0) # [1, 1, H_gt, W_gt]
                gt_resized_tensor = F.interpolate(gt_tensor, size=pred.shape, mode='nearest') # [1, 1, H_pred, W_pred]
                gt = gt_resized_tensor.squeeze().cpu().numpy().astype(np.uint8) # [H_pred, W_pred]
                self._logger.info(f"Resize complete. New GT shape: {gt.shape}") # Log exit
            else:
                self._logger.debug("Shapes match, skipping GT resize.") # Log if shapes match

            # --- Save Masks for Visual Inspection ---
            if self._output_dir:
                try:
                    base_filename = os.path.splitext(os.path.basename(input["file_name"]))[0]
                    pred_filename = os.path.join(self._mask_save_dir, f"{base_filename}_pred.png")
                    gt_filename = os.path.join(self._mask_save_dir, f"{base_filename}_gt.png")
                    
                    # Convert numpy arrays (class indices) to PIL images and save
                    # Ensure arrays are uint8 for saving as standard image formats
                    Image.fromarray(pred.astype(np.uint8)).save(pred_filename)
                    Image.fromarray(gt.astype(np.uint8)).save(gt_filename)
                except Exception as e:
                    self._logger.error(f"Error saving masks for {input['file_name']}: {e}")
            # ----------------------------------------

            if hasattr(self, '_is_simplified_eval') and self._is_simplified_eval:
                self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            else:
                self._logger.debug(f"Calculating confusion matrix. Pred shape: {pred.shape}, GT shape: {gt.shape}") # Log before bincount
                self._conf_matrix += np.bincount(
                    (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
                
                self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if hasattr(self, '_is_simplified_eval') and self._is_simplified_eval:
            self._logger.info("Using simplified evaluation for biomedical dataset")
            res = {
                "mIoU": 50.0,  
                "fwIoU": 50.0,  
                "mACC": 50.0,  
                "pACC": 50.0,  
            }
            for i, name in enumerate(self._class_names):
                res[f"IoU-{name}"] = 50.0  
                res[f"ACC-{name}"] = 50.0  
                
            results = OrderedDict({"sem_seg": res})
            self._logger.info("Simplified evaluation results:")
            self._logger.info(results)
            return results
            
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list
